
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_first = torch.nn.Conv2d(1, 2, (1, 1), stride=(1, 1))
        self.bn_first = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v0 = self.conv_first(x1)
        v1 = torch.relu(v0)
        v2 = self.bn_first(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
