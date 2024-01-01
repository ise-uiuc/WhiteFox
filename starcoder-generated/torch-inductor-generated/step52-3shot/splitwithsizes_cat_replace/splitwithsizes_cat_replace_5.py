
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.resnet_block_1 = ResBlock(
            conv2d = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            conv2d_0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            conv2d_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            bn = nn.BatchNorm2d(num_features=32),
            bn_0 = nn.BatchNorm2d(num_features=32),
            bn_1 = nn.BatchNorm2d(num_features=32),
            relu = nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(num_features=32, affine=False, track_running_stats=False),
            nn.ReLU(),
        )
        self.resnet_block_3 = ResBlock(
            conv2d = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d_0 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            conv2d_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            bn = nn.BatchNorm2d(num_features=64),
            bn_0 = nn.BatchNorm2d(num_features=64),
            bn_1 = nn.BatchNorm2d(num_features=64),
            relu = nn.ReLU(),
        )
    def forward(self, input):
        split_tensors = torch.split(input, [1, 1, 1], dim=1)
        x_2 = self.resnet_block_1(split_tensors[2])
        x_1 = self.resnet_block_1(split_tensors[1])
        x_0 = self.resnet_block_1(split_tensors[0])
        x_0 = cat([x_0, x_1, x_2], dim=1)
        split_tensors = torch.split(x_0, [1, 1, 1], dim=1)
        x_0 = self.resnet_block_3(split_tensors[0])
        split_tensors = torch.split(x_0, [1, 1, 1], dim=1)
        x_1 = self.block2(split_tensors[1])
        x_0 = self.block0(split_tensors[0])
        x_2 = self.resnet_block_3(split_tensors[2])
        x_0 = self.block2(split_tensors[0])
        x_1 = self.resnet_block_3(split_tensors[1])
        x_2 = self.block2(split_tensors[2])
        out = (torch.cat([x_0, x_1, x_2], dim=3), torch.split(input, [1, 1, 1], dim=1))
        return out
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
