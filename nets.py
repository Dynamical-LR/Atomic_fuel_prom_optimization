import torch
import torch.nn as nn
import torch.nn.functional as F

### целевой прототип

class AlphaZero(nn.Module):

    def __init__(self, ):
        super().__init__()

        self.conv0 = nn.Conv2d(6, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)

        self.bn0 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.mp0 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.layer2_conv = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer2_downsample = nn.Sequential(
            nn.Conv2d(32, 96, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.layer3_conv = nn.Sequential(
            nn.Conv2d(96, 288, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(288, 288, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer3_downsample = nn.Sequential(
            nn.Conv2d(96, 288, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.layer4_conv = nn.Sequential(
            nn.Conv2d(288, 864, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(864, 864, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False),
            nn.BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.layer4_downsample = nn.Sequential(
            nn.Conv2d(288, 864, kernel_size=(1, 3), stride=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(864, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(in_features=864, out_features=1, bias=True)

    def forward(self, x):
        x = self.mp0(self.relu(self.bn0(self.conv0(x))))

        x = self.layer1(x)

        # x1 = self.layer1(x)
        # x2 = self.layer1_downsample(x)
        # x = x1 + x2

        x1 = self.layer2_conv(x)
        x2 = self.layer2_downsample(x)
        x = x1 + x2

        x1 = self.layer3_conv(x)
        x2 = self.layer3_downsample(x)
        x = x1 + x2

        x1 = self.layer4_conv(x)
        x2 = self.layer4_downsample(x)
        x = x1 + x2

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x