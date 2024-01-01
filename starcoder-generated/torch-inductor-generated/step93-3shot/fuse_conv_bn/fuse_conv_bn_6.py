
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv = torch.nn.Conv3d(2,2,1)
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm3d(1)
        torch.manual_seed(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return y
# Inputs to the model
x = torch.randn(1, 2, 3, 3, 3)
