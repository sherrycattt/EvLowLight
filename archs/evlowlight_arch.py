from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from basicsr.archs.basicvsr_arch import ConvResidualBlocks
from basicsr.ops.dcn import ModulatedDeformConvPack
from basicsr.utils.registry import ARCH_REGISTRY
from kornia.filters import GaussianBlur2d
from torchvision.models.optical_flow._utils import make_coords_grid

from archs.arch_util import four_point_to_flow, autocast, CorrBlock
from archs.arch_util import interpolate_grid, warp_homo
from archs.evflow_arch import EventFlowNet
from archs.imflow_arch import FrameFlowNet


class LowresStreamNet(nn.Module):
    def __init__(self, lr_size=256, n_in=3, n_out=12,
                 channel_multiplier=1, luma_bins=8, spatial_bin=16, nl_local=2):
        super(LowresStreamNet, self).__init__()
        self.input_size = lr_size
        n_in = n_in
        n_out = n_out
        self.luma_bins = luma_bins
        self.spatial_bin = spatial_bin
        nc_base = 8 * channel_multiplier
        n_coarse = 4

        n_layers_splat = int(np.log2(lr_size / spatial_bin))
        splat_layers = []
        prev_ch = n_in
        for i in range(n_layers_splat):
            next_ch = channel_multiplier * (2 ** i) * luma_bins
            splat_layers.append(nn.Conv2d(prev_ch, next_ch, kernel_size=3, stride=2, padding=1))
            splat_layers.append(nn.ReLU())
            prev_ch = next_ch
        self.splat_features = nn.Sequential(*splat_layers)
        splat_ch = prev_ch

        n_global_down_layers = int(np.log2(spatial_bin / n_coarse))  # 4x4 at the coarsest lvl
        global_conv_layers = []
        for i in range(n_global_down_layers):
            next_ch = nc_base * luma_bins
            global_conv_layers.append(nn.Conv2d(prev_ch, next_ch, kernel_size=3, stride=2, padding=1))
            global_conv_layers.append(nn.ReLU())
            prev_ch = next_ch
        self.global_features_conv = nn.Sequential(*global_conv_layers)

        prev_ch = prev_ch * (n_coarse ** 2)
        self.global_features_fc = nn.Sequential(
            nn.Linear(prev_ch, 4 * nc_base * luma_bins),
            nn.ReLU(),
            nn.Linear(4 * nc_base * luma_bins, 2 * nc_base * luma_bins),
            nn.ReLU(),
            # don't normalize before fusion
            nn.Linear(2 * nc_base * luma_bins, 1 * nc_base * luma_bins),
        )

        prev_ch = splat_ch
        local_layers = []
        for i in range(nl_local):
            next_ch = nc_base * luma_bins
            local_layers.append(
                nn.Conv2d(
                    prev_ch, next_ch, kernel_size=3, padding=1,
                    bias=True if (i < (nl_local - 1)) else False,
                ), )
            if i < (nl_local - 1):
                local_layers.append(nn.ReLU())
            prev_ch = next_ch
        self.local_features = nn.Sequential(*local_layers)

        self.conv_fusion = nn.Conv2d(
            prev_ch, n_out * luma_bins, kernel_size=1)

    def forward(self, x):
        bs = x.size(0)  # in: bs * 3 * 256 * 256
        x = self.splat_features(x)  # out: bs * 64 * 16 * 16
        x_local = self.local_features(x)  # out: bs * 64 * 16 * 16
        x_global = self.global_features_conv(x)  # out: bs * 64 * 4 * 4
        x_global = self.global_features_fc(x_global.reshape(bs, -1))  # out: bs * 64
        fused = F.relu(x_global.unsqueeze(-1).unsqueeze(-1) + x_local)
        out = self.conv_fusion(fused)
        return out.view(bs, -1, self.luma_bins, self.spatial_bin, self.spatial_bin)  # out: bs * 64 * 16 * 16


class HighresStreamNet(nn.Module):
    def __init__(self, input_nc=3, complexity=16):
        super(HighresStreamNet, self).__init__()

        self.blur2d = GaussianBlur2d(kernel_size=(17, 17), sigma=(2., 2.))
        self.blur2d.requires_grad_(True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_nc, out_channels=complexity, kernel_size=1), nn.ReLU(),
            nn.Conv2d(in_channels=complexity, out_channels=1, kernel_size=1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.blur2d(x)
        return self.conv(x)


class ExposureEstimationNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, lr_size=256, mode='',
                 channel_multiplier=1, luma_bins=8, spatial_bin=16, guide_complexity=16, nl_local=2,
                 kernel_size=3, event_nc=15, lr_width=64, lr_depth=7, if_nn=False, num_features=0,
                 ):
        super(ExposureEstimationNet, self).__init__()
        self.num_inputs = input_nc
        self.input_size = lr_size
        self.kernel_size = kernel_size
        self.num_outputs = output_nc
        self.width = lr_width
        self.depth = lr_depth
        self.degree = 3
        self.num_params = 1

        self.lowres_stream = LowresStreamNet(
            lr_size=lr_size,
            n_in=input_nc + event_nc,
            n_out=self.num_params,
            channel_multiplier=channel_multiplier,
            luma_bins=luma_bins, spatial_bin=spatial_bin,
            nl_local=nl_local
        )
        self.guide_func = HighresStreamNet(input_nc=input_nc, complexity=guide_complexity)

    def forward(self, inputs_lqs, inputs_evs):
        guide_map = self.guide_func(inputs_lqs) * 2. - 0.5
        bs, d, h, w = guide_map.shape
        base_grid = torch.stack(torch.meshgrid([
            torch.linspace(0, w - 1, w),
            torch.linspace(0, h - 1, h)
        ], indexing="ij"), 2).permute(1, 0, 2).unsqueeze(dim=0).to(inputs_lqs)  # 1xHxWx2
        base_grid = (base_grid + 0.5) / torch.tensor([w, h]).to(inputs_lqs) * 2. - 0.5
        grid_hr = torch.cat([
            base_grid.repeat(bs, d, 1, 1, 1),
            guide_map.unsqueeze(-1)
        ], dim=-1)

        inputs_lr = F.interpolate(inputs_lqs, size=(self.input_size, self.input_size), mode='nearest')
        if inputs_evs.shape[-1] != self.input_size or inputs_evs.shape[-2] != self.input_size:
            inputs_evs = F.interpolate(inputs_evs, size=(self.input_size, self.input_size), mode='nearest')
        exposure_lr = self.lowres_stream(torch.cat([inputs_lr, inputs_evs], dim=1))

        exposure_hr = F.grid_sample(
            input=exposure_lr,
            grid=grid_hr,
            mode='bilinear',
            padding_mode="border",
            align_corners=False
        )
        return exposure_hr.view(bs, -1, h, w)


class GlobalUpdateLayer(nn.Module):
    def __init__(self, middle_dim=128):
        super(GlobalUpdateLayer, self).__init__()

        outputdim = middle_dim  # 164 = 32*5+4`
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        middle_dim = outputdim
        outputdim = middle_dim
        self.layer2 = nn.Sequential(nn.Conv2d(middle_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        middle_dim = outputdim
        outputdim = middle_dim
        self.layer3 = nn.Sequential(nn.Conv2d(middle_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        middle_dim = outputdim
        outputdim = middle_dim
        self.layer4 = nn.Sequential(nn.Conv2d(middle_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        middle_dim = outputdim
        outputdim_final = outputdim

        self.layer10 = nn.Sequential(nn.Conv2d(middle_dim, outputdim_final, 3, padding=1, stride=1),
                                     nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x


class GlobalUpdateBlock(nn.Module):
    def __init__(self, middle_dim=128):
        super().__init__()
        self.cnn = GlobalUpdateLayer(middle_dim)

    def forward(self, corr, flow):
        return self.cnn(torch.cat((corr, flow), dim=1))


class GlobalAlignmentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixed_precision = True
        self.update_block = GlobalUpdateBlock()  # sz = (32, 32)

    def forward(self, fmap_all_im, fmap_all_ev, iters_lev0=6):
        fmap_all_im = F.interpolate(fmap_all_im.view(-1, *fmap_all_im.size()[-3:]), (32, 32), mode='bilinear')
        fmap_all_ev = F.interpolate(fmap_all_ev.view(-1, *fmap_all_ev.size()[-3:]), (32, 32), mode='bilinear')
        batch_size = fmap_all_im.size(0)

        corr_fn = CorrBlock(num_levels=2, radius=4)
        corr_fn.build_pyramid(fmap_all_im, fmap_all_ev)

        coords0 = make_coords_grid(batch_size, 32, 32).to(fmap_all_im.device)
        coords1 = make_coords_grid(batch_size, 32, 32).to(fmap_all_im.device)

        four_point_disp = torch.zeros((batch_size, 2, 2, 2)).to(fmap_all_im.device)

        for itr in range(iters_lev0):
            corr = corr_fn.index_pyramid(coords1)
            with autocast(enabled=self.mixed_precision):
                delta_four_point = self.update_block(corr, coords1 - coords0)

            four_point_disp = four_point_disp + delta_four_point
            # downscale_factor are the scale between original size and fmap1
            coords1 = four_point_to_flow(four_point_disp, (batch_size, 32, 32))
        return four_point_disp


class FlowGuidedDeformConv(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(FlowGuidedDeformConv, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow_1, flow_2):  # x, warped_x, x_spatial, flows
        out = self.conv_offset(torch.cat([extra_feat, flow_1, flow_2], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset = offset + flow_2.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        # mask
        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                             self.dilation, mask)


@ARCH_REGISTRY.register()
class EvLowLightNet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 num_blocks=7,
                 ev_flow_factor=0.2,
                 prop_iter=1,
                 flow_iter=6,
                 eps=1e-6,
                 max_residue_magnitude=10,
                 ):

        super().__init__()
        self.mid_channels = mid_channels
        self.eps = eps
        self.ev_flow_factor = ev_flow_factor
        self.prop_iter = prop_iter
        self.flow_iter = flow_iter

        mid_channel_layers = (64, 96, 128, 256)

        # feature extraction module
        self.down1 = nn.Sequential(
            nn.Conv2d(3, mid_channel_layers[0], 3, 2, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.down2 = nn.Sequential(
            nn.Conv2d(mid_channel_layers[0], mid_channel_layers[1], 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.down3 = nn.Sequential(
            nn.Conv2d(mid_channel_layers[1], mid_channel_layers[2], 3, 2, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.mid1 = nn.Sequential(
            nn.Conv2d(mid_channel_layers[2], mid_channel_layers[3], 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.mid2 = ConvResidualBlocks(mid_channel_layers[3], mid_channels, 5)

        # optical flow
        self.ev_flow_net = EventFlowNet(flow_factor=ev_flow_factor)
        self.im_flow_net = FrameFlowNet()
        self.homo_net = GlobalAlignmentNet()

        self.mid_ev1 = ConvResidualBlocks(mid_channel_layers[3], mid_channels, 5)
        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.directions = ['backward', 'forward']
        i = 0
        for iter_ in range(self.prop_iter):
            for direc_ in self.directions:
                module = f'{direc_}_{iter_}'
                self.deform_align[module] = FlowGuidedDeformConv(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    padding=1,
                    deformable_groups=16,
                    max_residue_magnitude=max_residue_magnitude,
                )
                self.backbone[module] = ConvResidualBlocks((2 + i) * mid_channels, mid_channels, num_blocks)
                i += 1

        self.mid3 = ConvResidualBlocks((1 + i) * mid_channels, mid_channels, 5)

        # upsampling module
        self.mid4 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channel_layers[2], 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channel_layers[2] * 2, mid_channel_layers[1], 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channel_layers[1] * 2, mid_channel_layers[0], 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channel_layers[0] * 2, 64, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_last = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1), )
        self.illum_estimator = ExposureEstimationNet(input_nc=3, output_nc=3,
                                                     luma_bins=8, spatial_bin=16,
                                                     mode='noapl',
                                                     lr_size=128,
                                                     event_nc=15 * 2,  # 15*2
                                                     )

    def propagate(self, feats, feat_lqs, cond_lqs, cond_evs, flows, module_name):
        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, feat_lqs.size(1)))
        # mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feat_lqs[:, mapping_idx[idx], :, :, :]

            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]

                cond_n1 = cond_lqs[:, flow_idx[i], :, :, :]
                cond_ev = cond_evs[:, flow_idx[i], :, :, :]

                feat_prop = self.deform_align[module_name](
                    feat_prop,
                    torch.cat([cond_n1, cond_ev], dim=1), feat_current, flow_n1)

            # concatenate and residual blocks
            feat_prop = feat_prop + self.backbone[module_name](
                torch.cat([feat_current] + [feats[k][idx] for k in feats if k not in [module_name]] + [feat_prop],
                          dim=1)
            )
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def compute_flow(self, lqs, evs, fmap_all_im=None, feats_ev=None, feats_im=None):
        B, T, C, H, W = lqs.size()
        B, eT, eC, eH, eW = evs.size()

        evs_old = evs[:, :-1:2, :, :, :].reshape(-1, eC, eH, eW)  # for backwarp
        evs_new = evs[:, 1::2, :, :, :].reshape(-1, eC, eH, eW)  # for forwarp

        flows_down = {'backward': None, 'forward': None}
        flows_up = {'backward': None, 'forward': None}

        evs_1 = {'backward': evs_old, 'forward': evs_new.flip(-3).contiguous()}
        evs_2 = {'backward': evs_new, 'forward': evs_old.flip(-3).contiguous()}

        fmap1_ev = {'backward': None, 'forward': None}
        fmap2_ev = {'backward': None, 'forward': None}

        if feats_ev is None:
            feats_ev = {'backward': None, 'forward': None}

        feats_im = {
            'backward': feats_im[:, 1:, :, :, :].reshape(-1, *feats_im.size()[-3:]),
            'forward': feats_im[:, :-1, :, :, :].reshape(-1, *feats_im.size()[-3:])
        }

        if fmap_all_im is None:
            fmap_all_im = self.im_flow_net.feature_encoder(self.im_flow_net.preprocess(lqs).reshape(-1, C, H, W))

        fmap1_ev['backward'], fmap2_ev['backward'] = self.ev_flow_net.fnet([evs_1['backward'], evs_2['backward']])

        four_point_disp_all = self.homo_net(
            fmap_all_im,
            torch.cat([
                fmap1_ev['backward'].reshape(B, T - 1, -1, eH // 8, eW // 8),
                fmap2_ev['backward'].reshape(B, T - 1, -1, eH // 8, eW // 8)[:, -1:, :, :, :]
            ], dim=1).reshape(B * T, -1, eH // 8, eW // 8),
            iters_lev0=self.flow_iter)
        four_point_disp_all = four_point_disp_all.reshape(B, T, 2, 2, 2)
        four_point_disp = {
            'backward': four_point_disp_all[:, :-1, ...].reshape(-1, 2, 2, 2),
            'forward': four_point_disp_all[:, 1:, ...].reshape(-1, 2, 2, 2)
        }

        lqs = self.im_flow_net.preprocess(lqs)

        if fmap_all_im is None:
            fmap_all_im = self.im_flow_net.feature_encoder(lqs.reshape(-1, C, H, W))
        fmap_all_im = fmap_all_im.reshape(B, T, -1, H // 8, W // 8)
        fmap1_im = fmap_all_im[:, :-1, :, :, :].reshape(B * (T - 1), -1, H // 8, W // 8)
        fmap2_im = fmap_all_im[:, 1:, :, :, :].reshape(B * (T - 1), -1, H // 8, W // 8)

        lqs_context = {'backward': lqs[:, :-1, :, :, :].reshape(-1, C, H, W),
                       'forward': lqs[:, 1:, :, :, :].reshape(-1, C, H, W)}

        corr_block_im = CorrBlock(num_levels=self.im_flow_net.num_levels, radius=self.im_flow_net.radius)
        corr_block_im.build_pyramid(fmap1_im, fmap2_im)

        for direc_ in self.directions:
            if fmap1_ev[direc_] is None:
                fmap1_ev[direc_], fmap2_ev[direc_] = self.ev_flow_net.fnet([evs_1[direc_], evs_2[direc_]])

            corr_block_ev = CorrBlock(num_levels=self.ev_flow_net.num_levels, radius=self.ev_flow_net.radius)
            corr_block_ev.build_pyramid(fmap1_ev[direc_], fmap2_ev[direc_])

            flows_down[direc_], _ = self.ev_flow_net.compute(
                evs_2[direc_], corr_block=corr_block_ev, num_flow_updates=self.flow_iter, test_mode=True)
            flows_down[direc_] = interpolate_grid(flows_down[direc_], (H // 8, W // 8))

            if feats_ev[direc_] is None:
                feats_ev[direc_] = fmap1_ev[direc_]

            flows_down[direc_] = warp_homo(flows_down[direc_], four_point_disp[direc_], dsize=(H // 8, W // 8))

            feats_ev[direc_] = warp_homo(feats_ev[direc_], four_point_disp[direc_], dsize=(H // 8, W // 8))

            feats_ev[direc_] = self.mid_ev1(feats_ev[direc_])

            if 'forward' in direc_:
                corr_block_im.transpose_pyramid(fmap1_im.size())

            flows_down[direc_], _ = self.im_flow_net.compute(
                lqs_context[direc_],
                corr_block=corr_block_im,
                num_flow_updates=self.flow_iter,
                flow_init=flows_down[direc_],
                test_mode=True)

            feats_im[direc_] = corr_block_im.index_context(
                flow=flows_down[direc_].detach(),
                context_sup=feats_im[direc_],
            )

            feats_ev[direc_] = corr_block_im.index_context(
                flow=flows_down[direc_].detach(),
                context_sup=feats_ev[direc_],
                is_event=True
            )
            feats_im[direc_] = feats_im[direc_].reshape(B, T - 1, -1, H // 8, W // 8)
            feats_ev[direc_] = feats_ev[direc_].reshape(B, T - 1, -1, H // 8, W // 8)
            flows_down[direc_] = flows_down[direc_].view(B, T - 1, 2, H // 8, W // 8)

        evs = warp_homo(evs.view(B * (T - 1) * 2, -1, eH, eW),
                        torch.stack([v for k, v in four_point_disp.items()], dim=2).reshape(-1, 2, 2, 2),
                        dsize=(H, W)).reshape(B, (T - 1) * 2, -1, H, W)

        return flows_down, flows_up, feats_im, feats_ev, evs, four_point_disp_all

    def forward(self, lqs, evs, pretrain=False):
        if lqs.dim() == 4:
            lqs = lqs.unsqueeze(0)
        if evs.dim() == 4:
            evs = evs.unsqueeze(0)

        B, T, C, H, W = lqs.size()
        _, eT, eC, eH, eW = evs.size()

        # compute spatial features
        s1 = self.down1(lqs.view(-1, *lqs.size()[-3:]))  # 128*128
        s2 = self.down2(s1)  # 64*64
        s3 = self.down3(s2)  # 32*32
        s4 = self.mid1(s3)  # 32*32

        x = self.mid2(s4)
        feats_ev = None
        (flows_down, flows_up, feats_im, feats_ev, evs, four_point_disp) = self.compute_flow(
            lqs, evs,
            fmap_all_im=None,
            feats_ev=feats_ev,
            feats_im=x.view(B, T, -1, H // 8, W // 8))

        x = x.reshape(B, T, *x.size()[-3:])
        feats = OrderedDict()

        # feature propgation
        for iter_ in range(self.prop_iter):
            for direc_ in self.directions:
                module = f'{direc_}_{iter_}'

                feats[module] = []

                feats = self.propagate(
                    feats, feat_lqs=x,
                    cond_lqs=feats_im[direc_], cond_evs=feats_ev[direc_],
                    flows=flows_down[direc_], module_name=module)

        x = torch.cat([x] + [torch.stack(value, dim=1) for key, value in feats.items()], dim=-3)

        x = self.mid3(x.reshape(-1, *x.size()[-3:]))
        x = self.mid4(x)
        x = self.up1(torch.cat([x, s3], dim=-3))  # 64*64
        x = self.up2(torch.cat([x, s2], dim=-3))  # 128*128
        x = self.up3(torch.cat([x, s1], dim=-3))  # 256*256
        del s1, s2, s3
        torch.cuda.empty_cache()

        x = self.conv_last(x)
        x = x.reshape(*lqs.size())
        lqs_clean = x

        illums = []
        for i in range(T):
            illum = self.illum_estimator(
                inputs_lqs=lqs[:, i, :, :, :],
                inputs_evs=torch.cat([evs, evs[:, -2:, ...]], dim=1).reshape(B, T, 2 * eC, H, W)[:, i, :, :,
                           :],
            )
            illums.append(illum)
        illums = torch.stack(illums, dim=1)
        illums = torch.sigmoid(illums)

        return torch.pow(lqs_clean.clamp_min(1e-6), illums)
