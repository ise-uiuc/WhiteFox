
class ModelTanh(torch.nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 5, stride=1, padding=0, bias=False), torch.nn.BatchNorm2d(1), torch.nn.ReLU6(inplace=False))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(8, 1, 9, 9)
