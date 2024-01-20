import torch
from basicsr.models.video_recurrent_model import VideoBaseModel, VideoRecurrentModel
from basicsr.utils.registry import MODEL_REGISTRY

from archs.arch_util import ImagePadder


@MODEL_REGISTRY.register()
class EventVideoRecurrentTestModel(VideoRecurrentModel):
    def __init__(self, opt):
        VideoRecurrentModel.__init__(self, opt)
        self.center_frame_only = self.opt['datasets']['val'].get('center_frame_only', False)
        self.scale = self.opt['datasets']['val'].get('scale', 1)
        self.mod_size = self.opt['datasets']['val'].get('minimum_size', 1)
        self.im_padder = ImagePadder(min_size=self.mod_size)
        self.ev_padder = ImagePadder(min_size=self.mod_size // self.scale)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        assert len(self.lq.shape) == 5

        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        data['ev'].unsqueeze_(0)
        self.voxel = data['ev'].to(self.device)
        assert len(self.voxel.shape) == 5

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.center_frame_only:
            VideoBaseModel.dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        else:
            VideoRecurrentModel.dist_validation(self, dataloader, current_iter, tb_logger, save_img)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        self.lq = self.im_padder.pad(self.lq)
        self.voxel = self.ev_padder.pad(self.voxel)

        with torch.no_grad():
            self.output = self.net_g(self.lq, self.voxel)

        self.lq = self.im_padder.unpad(self.lq)
        self.voxel = self.ev_padder.unpad(self.voxel)
        self.output = self.im_padder.unpad(self.output)

        if self.center_frame_only and len(self.output.shape) > 4:
            self.output = self.output[:, n // 2, :, :, :]
            if hasattr(self, 'gt'):
                self.gt = self.gt.unsqueeze(1)

        self.net_g.train()
