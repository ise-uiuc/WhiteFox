
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 15, stride=1, padding=2)
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = self.conv1(v1)
        v3 = v2 - -30.4083
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 20, 20)
