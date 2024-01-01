
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1152, 1536, 1, stride=[1, 1])
        self.conv2 = torch.nn.Conv2d(1536, 1152, 1, stride=[1, 1])
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return self.conv2(v2)
# Inputs to the model
x = torch.randn(1, 1152, 8, 8)
