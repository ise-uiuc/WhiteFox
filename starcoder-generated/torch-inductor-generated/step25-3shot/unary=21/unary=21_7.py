
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 58, 14, stride=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        x2 = x1.permute(0, 2, 1)
        x3 = torch.tanh(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 1, 12, 16)
