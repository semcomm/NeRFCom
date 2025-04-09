import torch.nn as nn
import numpy as np
import os
import torch


class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = config.channel['type']
        self.chan_param = config.channel['chan_param']
        # self.device = config.device
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                config.channel['type'], config.channel['chan_param']))

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def forward(self, input, avg_pwr=None, power=1):
        if avg_pwr is None:
            avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        channel_in = channel_tx.reshape(-1)
        channel_in = channel_in[::2] + channel_in[1::2] * 1j
        channel_usage = channel_in.numel()
        channel_output = self.channel_forward(channel_in)
        channel_rx = torch.zeros_like(channel_tx.reshape(-1))
        channel_rx[::2] = torch.real(channel_output)
        channel_rx[1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma = np.sqrt(1.0 / (2 * 10 ** (self.chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output

    def forward_snr(self, input, snr_batch, avg_pwr_batch=None, power=1):
        B, L = input.shape

        if avg_pwr_batch is None:
            avg_pwr_batch = torch.mean(input ** 2, dim=1)
            channel_tx_batch = np.sqrt(power) * input / torch.sqrt(avg_pwr_batch.unsqueeze(-1) * 2)
        else:
            channel_tx_batch = np.sqrt(power) * input / torch.sqrt(avg_pwr_batch.unsqueeze(-1) * 2)

        input_shape = channel_tx_batch.shape # [B, L]
        channel_in_BL = channel_tx_batch.reshape(B, -1)
        channel_in = channel_in_BL[:, ::2] + channel_in_BL[:, 1::2] * 1j  # [B, L//2]

        nonzero_indices = torch.nonzero(channel_in)
        channel_usage = nonzero_indices.size(0)

        channel_output = self.channel_forward_snr(channel_in, snr_batch)
        channel_rx = torch.zeros_like(channel_tx_batch.reshape(B, -1))
        channel_rx[:, ::2] = torch.real(channel_output)
        channel_rx[:, 1::2] = torch.imag(channel_output)
        channel_rx = channel_rx.reshape(input_shape)

        return channel_rx * torch.sqrt(avg_pwr_batch.unsqueeze(-1) * 2), channel_usage

    def channel_forward_snr(self, channel_in, snr_batch):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            sigma_batch = [np.sqrt(1.0 / (2 * 10 ** (snr / 10))) for snr in snr_batch]
            chan_output = self.gaussian_noise_layer_snr(channel_tx, std_batch=sigma_batch)
            return chan_output

    def gaussian_noise_layer_snr(self, input_layer, std_batch):
        B, L = input_layer.shape
        device = input_layer.get_device()
        std_batch = torch.tensor(std_batch).reshape(B, 1).repeat(1, L)
        noise_real = torch.normal(mean=0.0, std=std_batch)
        noise_imag = torch.normal(mean=0.0, std=std_batch)
        noise = (noise_real + 1j * noise_imag).to(device)

        return input_layer + noise