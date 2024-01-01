
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, 1, stride=1, bias=False)
        self.tanh1 = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.tanh1(v1)
        v3 = self.tanh2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 400, 400)
