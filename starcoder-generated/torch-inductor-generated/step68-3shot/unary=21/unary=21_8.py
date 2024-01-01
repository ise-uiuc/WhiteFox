
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0, bias=True)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = torch.tanh(y1)
        return y2
# Inputs to the model
x = torch.randn(1, 32, 64, 128)
