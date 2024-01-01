
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3, 8, 2, dilation=2, stride=2, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256, 256)
