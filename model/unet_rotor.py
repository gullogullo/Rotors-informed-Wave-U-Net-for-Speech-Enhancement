import torch
import torch.nn as nn
import torch.nn.functional as F


class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7):
        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)


class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm1d(channel_out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, ipt):
        return self.main(ipt)


class DownSamplingRotors(nn.Module):
    def __init__(self, channel_in, channel_out, order, input_length, bias=False):
        super(DownSamplingRotors, self).__init__()
        kernel_size = 2 ** (order - 2)
        stride = 2 ** (order - 3)
        padding = kernel_size - stride - 1
        self.conv1 = nn.Conv1d(channel_in, channel_out, dilation=1, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.batch1 = nn.BatchNorm1d(channel_out)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.1)
        input_lstm = int(input_length / stride)
        self.lstm1 = nn.LSTM(input_size=input_lstm, hidden_size=128, batch_first=True)
        self.drop1 = nn.Dropout(p=0.2)
        output_linear = max(int(input_length / 2 ** order), 64)
        self.linear = nn.Linear(in_features=128, out_features=output_linear)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.leaky1(x)
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x = self.linear(x)
        return x


class Model(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24, input_length=16384):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]

        #          1    => 2    => 3    => 4    => 5    => 6   => 7   => 8   => 9  => 10 => 11 =>12
        # 16384 => 8192 => 4096 => 2048 => 1024 => 512 => 256 => 128 => 64 => 32 => 16 =>  8 => 4
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i]
                )
            )

        self.rotor_encoder = nn.ModuleList()
        self.rotor_encoder.append(
            nn.Sequential(
            nn.Conv1d(4, encoder_out_channels_list[0], dilation=1, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(encoder_out_channels_list[0]),
            nn.LeakyReLU(negative_slope=0.1)))
        self.rotor_encoder.append(
            DownSamplingRotors(channel_in=encoder_out_channels_list[0], 
                               channel_out=encoder_out_channels_list[4],
                               order=4,
                               input_length=input_length
                               )
        )
        self.rotor_encoder.append(
            DownSamplingRotors(channel_in=encoder_out_channels_list[4], 
                               channel_out=encoder_out_channels_list[8],
                               order=7,
                               input_length=input_length / 16
                               )
        )
        
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i]
                )
            )

        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input, rotors):
        tmp = []
        o = input
        rot = rotors

        # Down Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)
            if i % 4 == 0:
                j = int(i / 4)
                rot = self.rotor_encoder[j](rot)
                #if j != 0:
                #    o = o + rot
                o = o + rot
            tmp.append(o)
            # [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # [batch_size, T * 2, channels]
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # Skip Connection
            o = torch.cat([o, tmp[self.n_layers - i - 1]], dim=1)
            o = self.decoder[i](o)

        o = torch.cat([o, input], dim=1)
        o = self.out(o)
        return o