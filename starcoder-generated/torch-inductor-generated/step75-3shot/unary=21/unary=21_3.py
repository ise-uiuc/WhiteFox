
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 20)
        self.tanh = torch.nn.Tanh()
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x2 = torch.randn(1, 1, 9216, 4)
