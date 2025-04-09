import numpy as np
import torch
import os
import math
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# from compressai.ops import ste_round
# import compressai.ops.ops.quantize_ste as ste_round
# from compressai.ops.ops import quantize_ste as ste_round
from plenoxels.models.NTC_utils.layer.layers import Mlp
from plenoxels.models.NTC_utils.layer.analysis_transform import AnalysisTransform
from plenoxels.models.NTC_utils.layer.synthesis_transform import SynthesisTransform
from plenoxels.models.NTC_utils.loss.distortion import Distortion
from plenoxels.models.NTC_utils.utils import BCHW2BLN, BLN2BCHW
from plenoxels.models.NTC_utils.layer.jscc_encoder import JSCCEncoder
from plenoxels.models.NTC_utils.layer.jscc_decoder import JSCCDecoder
from plenoxels.models.NTC_utils.layer.channel import Channel

from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def ste_round(x: Tensor) -> Tensor:
    """
    Rounding with non-zero gradients. Gradients are approximated by replacing
    the derivative by the identity function.

    Used in `"Lossy Image Compression with Compressive Autoencoders"
    <https://arxiv.org/abs/1703.00395>`_

    .. note::

        Implemented with the pytorch `detach()` reparametrization trick:

        `x_round = x_round - x.detach() + x`
    """
    return torch.round(x) - x.detach() + x

class NTC_Hyperprior(nn.Module):
    def __init__(self, NTC_config):
        super().__init__()
        self.ga = AnalysisTransform(**NTC_config.ga_kwargs)
        self.gs = SynthesisTransform(**NTC_config.gs_kwargs)
        self.ha = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 5, stride=2, padding=2),
        )

        self.hs = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1)
        )

        # self.ha = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, stride=2, padding=1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, stride=2, padding=1),
        #     # nn.LeakyReLU(inplace=True),
        #     # nn.Conv2d(256, 256, 5, stride=2, padding=2),
        # )
        #
        # self.hs = nn.Sequential(
        #     nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(256, 512, 3, stride=2, padding=1, output_padding=1),
        #     # nn.ReLU(inplace=True),
        #     # nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1)
        # )
        
        self.entropy_bottleneck = EntropyBottleneck(256)
        self.gaussian_conditional = GaussianConditional(None)
        self.distortion = Distortion(NTC_config)
        self.H = self.W = 0

    def update_resolution(self, H, W):
        if H != self.H or W != self.W:
            self.ga.update_resolution(H, W)
            self.gs.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W

    def forward(self, input_image, require_probs=False):
        # print(f"NTC forward: ")
        # print(f"input_image [{torch.min(input_image):.4f}-{torch.max(input_image):.4f}] ")

        B, C, H, W = input_image.shape
        print(f"input_image.shape = {input_image.shape}") # torch.Size([3, 32, 512, 512])

        self.update_resolution(H, W)
        y = self.ga(input_image)
        print(f"y [{torch.min(y):.4f}-{torch.max(y):.4f}] {y.shape}") # torch.Size([3, 256, 32, 32])

        z = self.ha(y)
        # print(f"z.shape = {z.shape}")

        _, z_likelihoods = self.entropy_bottleneck(z)
        print(f"z_likelihoods [{torch.min(z_likelihoods):.4f}-{torch.max(z_likelihoods):.4f}] ")

        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        # print(f"z_hat.shape = {z_hat.shape}")
        #
        print(f"\nz [{torch.min(z):.4f}-{torch.max(z):.4f}] "
              f"\nz_offset [{torch.min(z_offset):.4f}-{torch.max(z_offset):.4f}] "
              f"\nz_tmp [{torch.min(z_tmp):.4f}-{torch.max(z_tmp):.4f}] "
              f"\nz_hat [{torch.min(z_hat):.4f}-{torch.max(z_hat):.4f}] ")

        gaussian_params = self.hs(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        # scales_hat = torch.clamp(scales_hat, min=1e-6)

        # print(f"scales_hat.shape={scales_hat.shape} means_hat.shape={means_hat.shape}")
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        print(f"y_likelihoods [{torch.min(y_likelihoods):.4f}-{torch.max(y_likelihoods):.4f}] {y_likelihoods.shape}")

        # breakpoint output
        # hy = torch.log(y_likelihoods) / -math.log(2)  # [B, 256, H/16, W/16]
        # hy = y  # [B, 256, H/16, W/16]
        # hy = y
        # print(f"hy={hy.shape}")
        # is_plot_heatmap = True
        # if is_plot_heatmap:
        #     heatmap_dir = f"./heatmap"
        #     if not os.path.exists(heatmap_dir):
        #         os.makedirs(heatmap_dir)
        #     self.plot_ref_heatmap(hy, heatmap_dir)
        #
        #     # self.plot_dist(hy, heatmap_dir)
        # exit()

        # print(f"\ngaussian_params [{torch.min(gaussian_params):.4f}-{torch.max(gaussian_params):.4f}] "
        #       f"\ny [{torch.min(y):.4f}-{torch.max(y):.4f}] "
        #     f"\nmeans_hat [{torch.min(means_hat):.4f}-{torch.max(means_hat):.4f}] "
        #     f"\nscales_hat [{torch.min(scales_hat):.4f}-{torch.max(scales_hat):.4f}] ")

        y_hat = ste_round(y - means_hat) + means_hat

        hy = y_hat
        # print(f"hy={hy.shape}")
        is_plot_heatmap = False
        if is_plot_heatmap:
            heatmap_dir = f"./heatmap"
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            self.plot_ref_heatmap(hy, heatmap_dir)

            # self.plot_dist(hy, heatmap_dir)
        # exit()



        x_hat = self.gs(y_hat)
        # print(f"input_image={input_image.shape}, x_hat={x_hat.shape}")
        mse_loss = self.distortion(input_image, x_hat)
        print(f"NTC y_likelihoods.sum = {torch.log(y_likelihoods).sum() / -math.log(2)}")
        bpp_y = torch.log(y_likelihoods).sum() / (-math.log(2) * H * W * C) / B
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * H * W * C) / B
        print(f"NTC y_likelihoods.sum = {torch.log(y_likelihoods).sum() / -math.log(2)} {bpp_y} {bpp_z}")

        if require_probs:
            return mse_loss, bpp_y, bpp_z, x_hat, y, y_likelihoods, scales_hat, means_hat
        else:
            return mse_loss, bpp_y, bpp_z, x_hat

    def plot_ref_heatmap(self, hy, heatmap_dir):
        # 索引第一个数据点并在通道上求和，形状将从 [32, 32, 32] 变为 [32, 32]
        plane_name = [['y', 'x'], ['y', 'z'], ['x', 'z']]

        fig = plt.figure(figsize=(18, 6))  # 创建一个足够宽的图来放置三个子图和一个色条
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], width_ratios=[1, 1, 1])  # 分配子图空间和色条空间

        for plane_id, hy_plane in enumerate(hy):
            hy_sum = torch.sum(hy_plane, dim=0)  # 在通道维度求和
            hy_data = hy_sum.detach().cpu().numpy().astype(np.uint8)  # 转换为 NumPy 数组
            hy_norm = hy_data / 256  # 归一化

            ax = fig.add_subplot(gs[0, plane_id])  # 为每个子图分配位置
            cax = ax.imshow(hy_norm, cmap='viridis', vmin=0, vmax=1)  # 使用相同的色谱和规范化
            ax.set_xticks([])  # 不显示x轴
            ax.set_yticks([])  # 不显示y轴
            ax.set_xlabel(plane_name[plane_id][0], fontsize=16)  # 显示x轴标签
            ax.set_ylabel(plane_name[plane_id][1], fontsize=16)  # 显示y轴标签

        # 添加共用的色条
        cbar_ax = fig.add_subplot(gs[1, :])  # 色条跨越所有列
        cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal')  # 设置色条为水平
        cbar.set_label("Normalized Bandwidth Allocation", fontsize=40)
        cbar.ax.tick_params(labelsize=34)

        plt.tight_layout()
        print(f"Saving to {heatmap_dir}/combined_heatmap.png")
        plt.savefig(f'{heatmap_dir}/combined_heatmap.png', bbox_inches='tight')  # 保存整个图形
        plt.close()  # 关闭图形，释放内存


    def plot_dist(self, hy, heatmap_dir):
        import matplotlib.pyplot as plt
        print(f"hy={hy.shape}")

        hy_sum = torch.sum(hy, dim=1)
        hy_sum = hy_sum.view(-1).cpu().numpy()
        plt.figure()

        # 绘制直方图
        plt.hist(hy_sum, bins=100)
        plt.savefig(f'{heatmap_dir}/hist.png', bbox_inches='tight')
        print(f"[{min(hy_sum)}, {max(hy_sum)}]")

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss


class NTSCC_Hyperprior(NTC_Hyperprior):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.channel = Channel(config)
        self.fe = JSCCEncoder(**config.fe_kwargs)
        self.fd = JSCCDecoder(**config.fd_kwargs)
        if config.use_side_info:
            embed_dim = config.fe_kwargs['embed_dim']
            self.hyprior_refinement = Mlp(embed_dim * 3, embed_dim * 6, embed_dim)
        self.eta = config.eta

    def feature_probs_based_Gaussian(self, feature, mean, sigma):
        sigma = sigma.clamp(1e-10, 1e10) if sigma.dtype == torch.float32 else sigma.clamp(1e-10, 1e4)
        gaussian = torch.distributions.normal.Normal(mean, sigma)
        prob = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
        likelihoods = torch.clamp(prob, 1e-10, 1e10)  # B C H W
        return likelihoods

    def update_resolution(self, H, W):
        # Update attention mask for W-MSA and SW-MSA
        if H != self.H or W != self.W:
            self.ga.update_resolution(H, W)
            self.fe.update_resolution(H // 16, W // 16)
            self.gs.update_resolution(H // 16, W // 16)
            self.fd.update_resolution(H // 16, W // 16)
            self.H = H
            self.W = W

    def forward(self, input_image, **kwargs):
        B, C, H, W = input_image.shape
        num_pixels = H * W * C
        self.update_resolution(H, W)

        # NTC forward
        mse_loss_ntc, bpp_y, bpp_z, x_hat_ntc, y, y_likelihoods, scales_hat, means_hat = \
            self.forward_NTC(input_image, require_probs=True)
        # print(f"y.shape={y.shape}, means_hat.shape={means_hat.shape}, scales_hat.shape={scales_hat.shape}")
        # print(f"NTSCC forward:"
        #       f"\ny [{torch.min(y):.4f}-{torch.max(y):.4f}] "
        #       f"\nmeans_hat [{torch.min(means_hat):.4f}-{torch.max(means_hat):.4f}] "
        #       f"\nscales_hat [{torch.min(scales_hat):.4f}-{torch.max(scales_hat):.4f}] ")

        y_likelihoods = self.feature_probs_based_Gaussian(y, means_hat, scales_hat)
        print(f"y_likelihoods [{torch.min(y_likelihoods)}-{torch.max(y_likelihoods)}]")

        # breakpoint output
        hy = torch.log(y_likelihoods) / -math.log(2)  # [B, 256, H/16, W/16]
        # hy = y
        print(f"hy={hy.shape}")
        is_plot_heatmap = True
        if is_plot_heatmap:
            heatmap_dir = f"./heatmap"
            if not os.path.exists(heatmap_dir):
                os.makedirs(heatmap_dir)
            self.plot_ref_heatmap(hy, heatmap_dir)
            # self.plot_dist(hy, heatmap_dir)
        exit()

        # DJSCC forward
        s_masked, mask_BCHW, indexes = self.fe(y, y_likelihoods.detach(), eta=self.eta)

        # Pass through the channel.
        mask_BCHW = mask_BCHW.byte()
        channel_input = torch.masked_select(s_masked, mask_BCHW)
        channel_output, channel_usage = self.channel.forward(channel_input)
        s_hat = torch.zeros_like(s_masked)
        s_hat[mask_BCHW] = channel_output
        cbr_y = channel_usage / num_pixels

        # Another realization of channel.
        # avg_pwr = torch.sum(s_masked ** 2) / mask_BCHW.sum()
        # s_hat, _ = self.channel.forward(s_masked, avg_pwr)
        # s_hat = s_hat * mask_BCHW
        # cbr_y = mask_BCHW.sum() / (B * num_pixels * 2)
        print(f"s_hat [{torch.min(s_hat)}-{torch.max(s_hat)}]")

        y_hat = self.fd(s_hat, indexes)
        print(f"y [{torch.min(y)}-{torch.max(y)}] // y_hat [{torch.min(y_hat)}-{torch.max(y_hat)}]")

        # hyperprior-aided decoder refinement (optional)
        if self.config.use_side_info:
            y_combine = torch.cat([BCHW2BLN(y_hat), BCHW2BLN(means_hat), BCHW2BLN(scales_hat)], dim=-1)
            y_hat = BLN2BCHW(BCHW2BLN(y_hat) + self.hyprior_refinement(y_combine), H // 16, W // 16)

        gs_out = self.gs(y_hat)
        print(f"gs_out [{torch.min(gs_out)}-{torch.max(gs_out)}]")
        # x_hat_ntscc = gs_out.clip(0, 1)
        x_hat_ntscc = gs_out + 0.5
        mse_loss_ntscc = self.distortion(input_image, x_hat_ntscc)

        return mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc


    def forward_NTC(self, input_image, **kwargs):
        return super(NTSCC_Hyperprior, self).forward(input_image, **kwargs)
