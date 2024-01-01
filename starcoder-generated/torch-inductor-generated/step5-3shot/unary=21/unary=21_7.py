
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 10, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 32, 128)
