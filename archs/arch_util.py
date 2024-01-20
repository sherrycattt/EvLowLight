import kornia.geometry as tgm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple, _ntuple
from torchvision.models.optical_flow._utils import make_coords_grid, grid_sample

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


def four_point_to_flow(four_point, sz, ori_sz=(32, 32), is_trans_params=False, is_inverse=False):
    four_point = four_point * torch.Tensor(
        [float(sz[-1] - 1) / float(ori_sz[-1] - 1), float(sz[-2] - 1) / float(ori_sz[-2] - 1)]
    ).to(four_point.device).view(1, 2, 1, 1)

    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([sz[-1] - 1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, sz[-2] - 1])
    four_point_org[:, 1, 1] = torch.Tensor([sz[-1] - 1, sz[-2] - 1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(sz[0], 1, 1, 1)
    four_point_new = four_point_org + four_point

    four_point_org = four_point_org.flatten(2).permute(0, 2, 1).contiguous()
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1).contiguous()
    if is_inverse:
        trans_10 = tgm.get_perspective_transform(four_point_new, four_point_org)
    else:
        trans_10 = tgm.get_perspective_transform(four_point_org, four_point_new)
    if is_trans_params:
        return trans_10
    gridy, gridx = torch.meshgrid(torch.linspace(0, sz[-1] - 1, steps=sz[-1]),
                                  torch.linspace(0, sz[-2] - 1, steps=sz[-2]), indexing="ij")
    points = torch.cat(
        (gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, sz[-1] * sz[-2]))),
        dim=0).unsqueeze(0).repeat(sz[0], 1, 1).to(four_point.device)
    points_new = trans_10.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    flow = torch.cat((points_new[:, 0, :].reshape(sz[0], sz[-1], sz[-2]).unsqueeze(1),
                      points_new[:, 1, :].reshape(sz[0], sz[-1], sz[-2]).unsqueeze(1)), dim=1)
    return flow


def warp_homo(x, four_point_disp, dsize, is_inverse=False):
    H = four_point_to_flow(four_point_disp, (four_point_disp.size(0), 1, *x.size()[-2:]), is_trans_params=True,
                           is_inverse=is_inverse)
    return tgm.warp_perspective(x, H, dsize)


def interpolate_grid(flow, size, mode='bilinear', align_corners=True):
    h, w = flow.shape[-2:]
    assert flow.shape[-3] == 2
    new_h, new_w = size
    return F.interpolate(flow, size=size, mode=mode, align_corners=align_corners) * torch.Tensor(
        [float(new_w - 1) / float(w - 1), float(new_h - 1) / float(h - 1)]
    ).to(flow.device).view(1, 2, 1, 1)


class ImagePadder(object):
    def __init__(self, min_size=64):
        self.min_size = min_size
        self.reset()

    def reset(self):
        self.pad_height = None
        self.pad_width = None

    def pad_to_closest_larger_multiple_of_minimum_size(self, size):
        return (self.min_size - size % self.min_size) % self.min_size

    def pad(self, image):
        height, width = image.shape[-2:]
        pad_height = (self.min_size - height % self.min_size) % self.min_size
        pad_width = (self.min_size - width % self.min_size) % self.min_size
        if self.pad_width is None:
            self.pad_height = pad_height
            self.pad_width = pad_width
        else:
            assert pad_height == self.pad_height and pad_width == self.pad_width
        if image.dim() == 5:
            padding = _ntuple(6)((self.pad_width, 0, self.pad_height, 0, 0, 0))
        else:
            padding = _quadruple((self.pad_width, 0, self.pad_height, 0))
        return F.pad(image, padding, 'reflect')

    def unpad(self, image):
        # --------------------------------------------------------------- #
        # Removes the padded rows & columns                               #
        # --------------------------------------------------------------- #
        return image[..., self.pad_height:, self.pad_width:]


class CorrBlock(nn.Module):
    def __init__(self, *, num_levels: int = 4, radius: int = 4):
        super().__init__()
        self.num_levels = num_levels
        self.radius = radius

        self.corr_pyramid = [torch.tensor(0)]  # useless, but torchscript is otherwise confused :')

        self.out_channels = num_levels * (2 * radius + 1) ** 2

    def build_pyramid(self, fmap1, fmap2):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """
        corr_volume = self._compute_corr_volume(fmap1, fmap2)

        b, h1, w1, c, h2, w2 = corr_volume.shape
        corr_volume = corr_volume.reshape(b * h1 * w1, c, h2, w2)
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def transpose_pyramid(self, shape):
        """Build the correlation pyramid from two feature maps.

        The correlation volume is first computed as the dot product of each pair (pixel_in_fmap1, pixel_in_fmap2)
        The last 2 dimensions of the correlation volume are then pooled num_levels times at different resolutions
        to build the correlation pyramid.
        """
        batch_size, _, h1, w1 = shape
        corr_volume = self.corr_pyramid[0].reshape(batch_size, h1 * w1, 1, -1).transpose(1, 3)

        corr_volume = corr_volume.reshape(-1, 1, h1, w1)
        # del self.corr_pyramid
        # torch.cuda.empty_cache()
        self.corr_pyramid = [corr_volume]
        for _ in range(self.num_levels - 1):
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr_volume)

    def index_pyramid(self, centroids_coords):
        """Return correlation features by indexing from the pyramid."""
        side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        di = torch.linspace(-self.radius, self.radius, side_len)
        dj = torch.linspace(-self.radius, self.radius, side_len)
        delta = torch.stack(torch.meshgrid(di, dj, indexing="ij"), dim=-1).to(centroids_coords.device)
        delta = delta.view(1, side_len, side_len, 2)

        b, _, h1, w1 = centroids_coords.shape  # _ = 2, shape: h1 * w1; index: h2 * w2
        centroids_coords_lvl = centroids_coords.permute(0, 2, 3, 1).reshape(b * h1 * w1, 1, 1, 2)

        indexed_pyramid = []
        for corr_volume in self.corr_pyramid:
            indexed_corr_volume = grid_sample(
                corr_volume,  # corr_volume: b * h1 * w1, 1, h2, w2
                centroids_coords_lvl + delta,  # (b * h1 * w1, side_len, side_len, 2)
                align_corners=True, mode="bilinear",
                # (b * h1 * w1, 1, side_len, side_len)
            ).view(b, h1, w1, side_len ** 2)
            indexed_pyramid.append(indexed_corr_volume)
            centroids_coords_lvl /= 2

        corr_features = torch.cat(
            indexed_pyramid, dim=-1).permute(0, 3, 1, 2).contiguous().view(b, self.out_channels, h1, w1)

        return corr_features

    def index_context(self, flow, context_sup, is_event=False):
        """Return correlation features by indexing from the pyramid."""
        side_len = 2 * self.radius + 1  # see note in __init__ about out_channels
        b, _, h1, w1 = flow.shape  # _ = 2, shape: h1 * w1; index: h2 * w2
        coords0 = make_coords_grid(b, h1, w1).to(flow.device).permute(0, 2, 3, 1)
        coords0 = (coords0.float() / torch.Tensor([float(w1 - 1), float(h1 - 1)]).to(flow.device) * 2 - 1).reshape(
            b * h1 * w1, 1, 1, 2)

        flow_grid = (flow.permute(0, 2, 3, 1) / torch.Tensor(
            [float(w1 - 1), float(h1 - 1)]).to(flow.device) * 2 - 1).reshape(b * h1 * w1, 1, 1, 2)

        delta_grid = torch.stack(
            torch.meshgrid(torch.linspace(0, 2, side_len), torch.linspace(0, 2, side_len), indexing="ij")[::-1], dim=-1
        ).float().to(flow.device).view(side_len ** 2, 1, 2)
        if is_event:
            c = context_sup.size(1)
            scaling = torch.linspace(0, 1, c + 2)[1:-1].view(1, c, 1).to(flow.device)
        else:
            c = 1
            scaling = torch.Tensor([1.]).view(1, c, 1).to(flow.device)
        sampling_point = coords0 + delta_grid * flow_grid * scaling

        corr_features = F.grid_sample(
            self.corr_pyramid[0],  # b * h1 * w1, 1, h2, w2
            sampling_point,  # (b * h1 * w1, side_len**2, c, 2)
            align_corners=True, mode="bilinear",
            # (b * h1 * w1, 1, side_len, side_len)
        ).view(b, h1, w1, side_len ** 2, c).permute(0, 3, 4, 1, 2).contiguous()

        context_features = F.grid_sample(
            context_sup.view(b * c, -1, *context_sup.size()[-2:]),  # b * c, 1, h1 * w1
            sampling_point.view(
                b, h1 * w1, side_len ** 2, c, 2).permute(0, 3, 1, 2, 4).contiguous().view(b * c, h1 * w1, side_len ** 2,
                                                                                          2),  # b * c, 1, h1 * w1,
            align_corners=True, mode="bilinear",
            #  (b*c, -1, h1 * w1, side_len ** 2)
        ).view(b, -1, h1, w1, side_len ** 2).permute(0, 4, 1, 2, 3).contiguous()

        context_features = (context_features * F.softmax(corr_features, dim=1)).sum(dim=1).view(b, -1, h1, w1).float()

        return context_features

    def _compute_corr_volume(self, fmap1, fmap2):
        batch_size, num_channels, h1, w1 = fmap1.shape
        _, _, h2, w2 = fmap2.shape
        fmap1 = fmap1.view(batch_size, num_channels, h1 * w1)
        fmap2 = fmap2.view(batch_size, num_channels, h2 * w2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch_size, h1, w1, 1, h2, w2)
        return corr / torch.sqrt(torch.tensor(num_channels))
