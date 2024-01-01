
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 513, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
