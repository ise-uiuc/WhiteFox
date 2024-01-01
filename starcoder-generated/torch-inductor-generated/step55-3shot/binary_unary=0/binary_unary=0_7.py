
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(64, 64)
    def forward(self, x):
        v1 = self.pooling(x)
        _, _, v2, _ = v1.shape
        v3 = self.flatten(v1)
        v4 = self.linear(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 256, 64, 128)
