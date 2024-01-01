
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = lambda x: x
        self.conv2 = lambda x: x
        self.sigmoid = lambda x: x
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
