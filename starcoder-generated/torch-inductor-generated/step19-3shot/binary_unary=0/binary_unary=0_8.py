
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(16, 16)
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = x + self.dense(x)
        v2 = v1 + self.dense(x)
        v3 = v2 + self.conv(x)
        return v3
# Inputs to the model
x = torch.randn(1, 16)
