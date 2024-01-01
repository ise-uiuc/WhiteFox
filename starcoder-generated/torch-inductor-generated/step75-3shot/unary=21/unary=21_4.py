
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, (1, 1), stride=(1, 1), padding=(0, 0), groups=128, bias=False)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 128, 16, 16)
