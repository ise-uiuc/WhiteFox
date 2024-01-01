
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=1, padding=0, dilation=1, bias=True)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = F.tanh(v1)
        return v2
# Inputs to the model
x = torch.rand(1, 4, 64, 64)
