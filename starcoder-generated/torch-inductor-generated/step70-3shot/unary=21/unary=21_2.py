
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(13, 16, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, input):
        v0 = self.conv2d(input)
        v1 = self.tanh(v0)
        return v1
# Inputs
input = torch.randn(1, 13, 28, 28)
