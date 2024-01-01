
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=2, bias=True)
        self.conv2 = torch.nn.Conv2d(16, 2, 2, padding=1, bias=True)
        self.tanh = torch.nn.Tanh()
    def forward(self, x17):
        v1 = self.conv1(x17)
        v2 = self.conv2(v1)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x17 = torch.randn(1, 3, 112, 112)
