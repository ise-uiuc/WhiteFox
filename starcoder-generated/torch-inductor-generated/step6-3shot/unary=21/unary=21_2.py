
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 2, stride=(3, 2), padding=(5, 1), dilation=(3, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 3, 96, 240)
