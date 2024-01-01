
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=1):
        z1 = torch.randperm(x1.size()[2] * x1.size()[3])
        x1 = torch.reshape(x1, shape=(-1, x1.size()[3], x1.size()[4]))
        x1 = torch.index_select(x1, dim=1, index=z1)
        x1 = torch.reshape(x1, shape=(-1, x1.size()[2], x1.size()[3]))
        v1 = self.conv(x1)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 4, 16, 16)
