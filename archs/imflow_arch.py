# Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow._utils import make_coords_grid
from torchvision.models.optical_flow.raft import (
    ResidualBlock, FeatureEncoder, MotionEncoder,
    MaskPredictor, RecurrentBlock, UpdateBlock, FlowHead)

from archs.arch_util import autocast, CorrBlock


class FrameFlowNet(nn.Module):
    def __init__(
            self,
            # Feature encoder
            feature_encoder_layers=(64, 64, 96, 128, 256),
            feature_encoder_block=ResidualBlock,
            feature_encoder_norm_layer=nn.InstanceNorm2d,
            # Context encoder
            context_encoder_layers=(64, 64, 96, 128, 256),
            context_encoder_block=ResidualBlock,
            context_encoder_norm_layer=nn.BatchNorm2d,
            # Correlation block
            corr_block_num_levels=4,
            corr_block_radius=4,
            # Motion encoder
            motion_encoder_corr_layers=(256, 192),
            motion_encoder_flow_layers=(128, 64),
            motion_encoder_out_channels=128,
            # Recurrent block
            recurrent_block_hidden_state_size=128,
            recurrent_block_kernel_size=((1, 5), (5, 1)),
            recurrent_block_padding=((0, 2), (2, 0)),
            # Flow head
            flow_head_hidden_size=256,
    ):
        super(FrameFlowNet, self).__init__()
        self.mixed_precision = True
        self.num_levels = corr_block_num_levels
        self.radius = corr_block_radius
        # feature network, context network, and update block
        self.feature_encoder = FeatureEncoder(
            block=feature_encoder_block, layers=feature_encoder_layers, norm_layer=feature_encoder_norm_layer
        )
        self.context_encoder = FeatureEncoder(
            block=context_encoder_block, layers=context_encoder_layers, norm_layer=context_encoder_norm_layer
        )

        motion_encoder = MotionEncoder(
            in_channels_corr=self.num_levels * (2 * self.radius + 1) ** 2,
            corr_layers=motion_encoder_corr_layers,
            flow_layers=motion_encoder_flow_layers,
            out_channels=motion_encoder_out_channels,
        )
        # See comments in forward pass of RAFT class about why we split the output of the context encoder
        self.hidden_dim = recurrent_block_hidden_state_size
        self.context_dim = context_encoder_layers[-1] - self.hidden_dim
        recurrent_block = RecurrentBlock(
            input_size=motion_encoder_out_channels + self.context_dim,
            hidden_size=self.hidden_dim,
            kernel_size=recurrent_block_kernel_size,
            padding=recurrent_block_padding,
        )
        flow_head = FlowHead(in_channels=self.hidden_dim, hidden_size=flow_head_hidden_size)

        self.update_block = UpdateBlock(motion_encoder=motion_encoder, recurrent_block=recurrent_block,
                                        flow_head=flow_head)

        self.mask_predictor = MaskPredictor(
            in_channels=self.hidden_dim,
            hidden_size=256,
            multiplier=0.25,  # See comment in MaskPredictor about this
        )

        if not hasattr(self.update_block, "hidden_state_size"):
            raise ValueError("The update_block parameter should expose a 'hidden_state_size' attribute.")

        del self.mask_predictor
        torch.cuda.empty_cache()
        self.register_buffer('mean', torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def compute(self, image1, corr_block, num_flow_updates: int = 12, flow_init=None, test_mode=False):

        with autocast(enabled=self.mixed_precision):
            context_out = self.context_encoder(image1)
            # As in the original paper, the actual output of the context encoder is split in 2 parts:
            # - one part is used to initialize the hidden state of the recurent units of the update block
            # - the rest is the "actual" context.
            hidden_state, context = torch.split(context_out, [self.hidden_dim, self.context_dim], dim=1)
            hidden_state = torch.tanh(hidden_state)
            context = F.relu(context)

        return self.compute_flow(hidden_state, context, image1, corr_block, num_flow_updates, flow_init, test_mode)

    def compute_flow(self, hidden_state, context, image1, corr_block, num_flow_updates: int = 12, flow_init=None,
                     test_mode=False):
        batch_size, _, h, w = image1.shape
        coords0 = make_coords_grid(batch_size, h // 8, w // 8).to(image1.device)
        coords1 = make_coords_grid(batch_size, h // 8, w // 8).to(image1.device)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(num_flow_updates):
            coords1 = coords1.detach()  # Don't backpropagate gradients through this branch, see paper
            corr_features = corr_block.index_pyramid(centroids_coords=coords1)

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                hidden_state, delta_flow = self.update_block(hidden_state, context, corr_features, flow)

                coords1 = coords1 + delta_flow

                flow_predictions.append(coords1 - coords0)
        if test_mode:
            return coords1 - coords0, flow_predictions[-1]
        else:
            return flow_predictions

    def forward(self, image1, image2, num_flow_updates: int = 12, flow_init=None, test_mode=False):
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        with autocast(enabled=self.mixed_precision):
            fmaps = self.feature_encoder(torch.cat([image1, image2], dim=0))
            fmap1, fmap2 = torch.chunk(fmaps, chunks=2, dim=0)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_block = CorrBlock(num_levels=self.num_levels, radius=self.radius)
        corr_block.build_pyramid(fmap1, fmap2)

        return self.compute(image1, corr_block=corr_block, num_flow_updates=num_flow_updates, flow_init=flow_init,
                            test_mode=test_mode)
