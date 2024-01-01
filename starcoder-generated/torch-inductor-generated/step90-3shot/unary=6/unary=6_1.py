
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.pool2d = torch.nn.MaxPool2d(3, stride=3, padding=1)
        self.flatten = torch.flatten
        self.dense = torch.nn.Linear(16368, 1, bias=True)
    def forward(self, x2):
        v1 = self.conv(x2)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        v6 = self.pool2d(v5)
        v7 = self.flatten(v6, start_dim=1)
        v8 = self.dense(v7)
        return v8
# Inputs to the model
x2 = torch.randn(1, 3, 512, 256)
