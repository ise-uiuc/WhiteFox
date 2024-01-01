
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(0, 0))
    def forward(self, x24):
        v4 = self.conv(x24)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x24 = torch.randn(1, 3, 56, 56)
