
class Model(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Union[int, Tuple[int, int], List[int]], out_channels: int, groups: int, depth: int, kernel_size: int = 3, bias: bool = False, BatchNorm2d: nn.BatchNorm2d = nn.BatchNorm2d):
        super().__init__()
        hidden_channels_tuple = (hidden_channels if isinstance(hidden_channels, tuple) else (hidden_channels, hidden_channels))
        hidden_channels = hidden_channels_tuple[0]
        self.pools = [nn.AvgPool2d(kernel_size=2, stride=2)]
        self.blocks = nn.ModuleList([nn.Conv2d(in_channels=in_channels if i == 0 else hidden_channels, out_channels=hidden_channels, groups=groups, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=bias) for i in range(depth)])
        self.pools.append(nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.bottleneck_blocks = nn.ModuleList([nn.Conv2d(in_channels, hidden_channels_tuple[1], 1, bias=False), BatchNorm2d(hidden_channels_tuple[1]), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels_tuple[1], in_channels, 1, bias=False), BatchNorm2d(in_channels)])
        self.conv = nn.Conv2d(in_channels, in_channels, 1, groups=in_channels, bias=False)
 
    def forward(self, x):
        y = x
        for pool in self.pools:
            y = pool(y)
        y = self.blocks[0](y)
        for block in self.blocks[1:]:
            y = block(y)
        y = self.conv(y)
        for bottleneck_block in self.bottleneck_blocks:
            y = bottleneck_block(y)
        y += x
        y = self.fc(y.squeeze())
        return y

# Initializing the model
m = Model(3, 6, 10, 3, 2)

# Inputs to the model
x = torch.randn(1, m.in_channels, 28, 28)
