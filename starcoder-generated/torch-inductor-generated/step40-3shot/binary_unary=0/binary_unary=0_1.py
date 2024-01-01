
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(30, 30, (3, 3), stride=(1, 1), padding=(1, 1), groups=30)
    def forward(self, x):
        v1 = self.depthwise(x)
        v2 = self.depthwise(v1)
        v3 = torch.relu(v2)
        v4 = v1 + v3
        return v4 * v1 + 643.5
# Inputs to the model
x = torch.randn(1, 30, 51, 51)
