
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 1, stride=1, padding=1)
        self.dense = torch.nn.Linear(16, 8)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.flatten(v1, -1)
        v3 = self.dense(v2)
        v4 = torch.flatten(v3, 0)
        v5 = v4 - torch.tensor([.59,.55,.94,.37,.63])
        return v5
# Inputs to the model
x = torch.randn(1, 5, 32, 64)
