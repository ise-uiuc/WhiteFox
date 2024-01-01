
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(512, 1024, 9, stride=1, padding=0)
        self.pool1 = torch.nn.AvgPool1d(32, stride=8, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.pool1(v6)
        return v7

# Inputs to the model
x1 = torch.randn(1, 512, 1000)

