
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), bias=False)
        self.bn = torch.nn.BatchNorm1d(num_features=9216)
        self.tanh = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.conv_1(x0)
        x1 = torch.tanh(x1)
        x1 = torch.sum(x1, dim=2, keepdim=False)
        x2 = self.bn(x1)
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x0 = torch.randn(1, 128, 64, 1).repeat(1, 1, 1, 64)
