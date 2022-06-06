import torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess as P

class UnmuteTranscoder(nn.Module):
    def __init__(self, net_out):
        super().__init__()
        channel_cnt = [
            P.window_size * P.channel_cnt, # input
            32, 32, 32,
            64, 64,
            128, 128, 128, 128
        ]
        cnn_out_size = P.face_cols * P.face_rows // (P.pool_size ** (5 * 2)) * channel_cnt[-1]
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=channel_cnt[i-1],
                out_channels=channel_cnt[i],
                kernel_size=(3, 3),
                padding='same',
            )
            for i in range(1, len(channel_cnt))
        ])
        self.dense_layers = nn.ModuleList([
            nn.Linear(cnn_out_size, 512),
            nn.Linear(512, 512),
            nn.Linear(512, net_out),
        ])
        self.batch_norm = nn.ModuleList([
            nn.BatchNorm2d(c) for c in channel_cnt[1:]
        ])
        self.batch_norm2 = nn.ModuleList([
            nn.BatchNorm1d(512),
            nn.BatchNorm1d(512),
        ])
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        from itertools import chain
        # Is this needed?
        for c in chain(self.conv_layers, self.dense_layers):
            torch.nn.init.kaiming_normal_(c.weight)

    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = self.batch_norm[i](x)
            if i == len(self.conv_layers) - 1:
                x = torch.tanh(x)
            else:
                x = F.leaky_relu(x)
            if i % 2 == 0:
                x = F.max_pool2d(x, (P.pool_size, P.pool_size))
            if i > 0:
                if i < len(self.conv_layers) - 1:
                    x = self.dropout1(x)
                else:
                    x = self.dropout2(x)
        x = x.flatten(start_dim=1)
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            if i < len(self.batch_norm2):
                x = self.batch_norm2[i](x)
            if i == 0:
                x = torch.tanh(x)
                x = self.dropout2(x)
        return x
