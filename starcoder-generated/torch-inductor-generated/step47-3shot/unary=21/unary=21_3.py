
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2, stride=2)
        self.conv2 = torch.nn.Conv2d(2, 2, 2, stride=2)
    def forward(self, x):
        v2 = self.conv(x)
        v3 = torch.tanh(v2)
        return self.conv2(v3).squeeze(3)
# Inputs to the model
x = torch.randn(128, 2, 77, 87)
