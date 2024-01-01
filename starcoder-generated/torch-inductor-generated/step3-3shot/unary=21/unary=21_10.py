
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x5925):
        x5926 = self.conv(x5925)
        x5927 = torch.tanh(x5926)
        return x5927
# Inputs to the model
x5925 = torch.randn(1, 1, 256, 256)
