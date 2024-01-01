
class Model(torch.nn.Module):
    def __init__(self, min, max=1.0):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(1, 5, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        x = torch.tanh(self.conv2d(x))
        v2 = torch.clamp(x, min=self.min, max=self.max)
        return v2
min = 1
max = 1
# Inputs to the model
x=torch.randn(1, 1, 40, 64)
