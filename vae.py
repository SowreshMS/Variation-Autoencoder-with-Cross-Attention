import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.prune as prune

class Resnet(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.nonLinear1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.nonLinear2 = nn.SiLU()
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def forward(self, input):
        x = input
        x = self.norm1(x)
        x = self.nonLinear1(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonLinear2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        input = self.shortcut(input)
        return x + input

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.resnets = nn.ModuleList([])

        in_channel = 128
        out_channel = 128
        num_layers = 4
        for i in range(num_layers):
            if i != 0:
                out_channel = in_channel * 2
            self.resnets.append(Resnet(in_channel, out_channel))
            if i != num_layers - 1:
                self.resnets.append(nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=1))
            if i != 0:
                in_channel *= 2

        self.conv_norm_out = nn.GroupNorm(32, 1024, eps=1e-06, affine=True)
        self.conv_act  = nn.SiLU()
        self.conv_out = nn.Conv2d(1024, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv_in_decoder = nn.Conv2d(4, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.resnets_decoder = nn.ModuleList([])

        in_channel = 1024
        out_channel = 1024
        num_layers = 4
        for i in range(num_layers):
            if i != 0:
                out_channel = in_channel // 2
            self.resnets_decoder.append(Resnet(in_channel, out_channel))
            if i != num_layers - 1:
                self.resnets_decoder.append(nn.ConvTranspose2d(out_channel, out_channel, kernel_size=(2, 2), stride=(2, 2)))
            if i != 0:
                in_channel //= 2

        self.conv_norm_out_decoder = nn.GroupNorm(32, 128, eps=1e-06, affine=True)
        self.conv_act_decoder  = nn.SiLU()
        self.conv_out_decoder = nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


    def encoder(self, x):
        x = self.conv_in(x)

        for block in self.resnets:
            x = block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


    def decoder(self, x):
        x = self.conv_in_decoder(x)

        for block in self.resnets_decoder:
            x = block(x)

        x = self.conv_norm_out_decoder(x)
        x = self.conv_act_decoder(x)
        x = self.conv_out_decoder(x)

        return x

    def forward(self, x):
        mu, log_var = torch.chunk(self.encoder(x), 2, dim=1)
        return self.decoder(self.reparameterize(mu, log_var)), mu, log_var

def kl_loss(x_recon, x, mu, log_var):
    recon_loss = nn.functional.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence


