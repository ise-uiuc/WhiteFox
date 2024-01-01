
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(22, 128, 3, stride=1, padding=0, bias=False)
    def forward(self, x):
        v3 = self.conv(x)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 22, 299, 299)
