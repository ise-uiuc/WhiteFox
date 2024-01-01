
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(25, 8, 3, padding=(1, 1), stride=(2, 2))
    def forward(self, x18):
        x19 = self.conv(x18)
        x20 = torch.tanh(x19)
        return x20
# Inputs to the model
x18 = torch.randn(1, 25, 128, 128)
