import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(DQN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=output_shape)
        )

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape)) # 1 is the position of BATCH_SIZE
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv1(x).view(x.size(0), -1) # x.size(0) is BATCH_SIZE
        return self.fc(conv_out)