
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1024, 1536, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1536, 1536, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = torch.tanh(x1)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        v4 = torch.exp(v3)
        v5 = self.conv2(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1024, 16, 16)
