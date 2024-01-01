
class ModelTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(1, 1, 1)
        self.tanh = nn.Tanh()
    def forward(self, x1):
        x2 = self.conv1(x1)
        y2 = self.tanh(x2)
        return y2
# Inputs to the model

x1 = torch.randn(1, 1, 3, 3)
