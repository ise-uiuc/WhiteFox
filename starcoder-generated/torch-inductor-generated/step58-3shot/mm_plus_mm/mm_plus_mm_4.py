
class Model(nn.Module):
    def __init__(self, out_size=1, in_size=64):
        super(Model, self).__init__()
        self.block_list_1 = nn.ModuleList(get_blocks(ResidualBlock, 16, 2))
        self.block_list_2 = nn.ModuleList(get_blocks(ResidualBlock, 16, 2))
        self.linear = nn.Linear(1024, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(in_size, out_size)
    def forward(self, tensor1, tensor2, tensor3, tensor4, tensor5):
        x = torch.cat([tensor1, tensor2], 1)
        for _ in range(8):
            x = self.block_list_1[0](x)
        x = self.block_list_1[1](x)
        x1 = x
        for _ in range(8):
            x = self.block_list_2[0](x)
        x = self.block_list_2[1](x)
        x2 = x
        x = nn.AdaptiveAvgPool2d(1)(torch.add(x1,x2))
        x = x.view(-1, 4096)
        x = F.relu(self.bn1(self.linear(x)))
        x = self.bn2(x)
        x = torch.sigmoid(self.out(x))
        return x
# Inputs to the model
tensor1 = torch.randn(64, 32, 32, 3)
tensor2 = torch.randn(64, 32, 32, 3)
tensor3 = torch.randn(64, 32, 32, 3)
tensor4 = torch.randn(64, 32, 32, 3)
tensor4 = torch.randn(64, 32, 32, 3)
