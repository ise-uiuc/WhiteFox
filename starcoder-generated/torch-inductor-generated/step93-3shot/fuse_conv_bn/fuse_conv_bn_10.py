
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 1, 3, stride=(1, 2, 2), padding=(0, 1, 1), dilation=3, groups=2)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(1, affine=False)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 5, 5, 5)
