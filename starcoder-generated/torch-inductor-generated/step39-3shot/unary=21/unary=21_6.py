
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 18, 4, padding=1)
    def forward(self, x9):
        v10 = self.conv(x9)
        v11 = F.tanh(v10)
        return v11
# Inputs to the model
x9 = torch.randn(1, 3, 15, 15)
