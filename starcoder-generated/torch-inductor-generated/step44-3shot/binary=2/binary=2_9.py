
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(16, 43)
        self.conv = torch.nn.Conv2d(28, 10, 3, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = F.relu(self.dense(x1))
        v2 = self.conv(x2)
        v3 = v1 - v2
        v4 = v3 - 0.23
        return v4
# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 28, 6, 6)
