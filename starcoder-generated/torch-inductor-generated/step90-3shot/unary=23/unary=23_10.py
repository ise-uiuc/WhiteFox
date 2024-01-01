
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2048, 1280, 3, stride=[1], padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2048, 1, 1)
