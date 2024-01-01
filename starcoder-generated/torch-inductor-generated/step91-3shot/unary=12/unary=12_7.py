
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.flatten = torch.nn.Flatten()
        self.linear_ = torch.nn.Linear(64, 128)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.flatten(v1)
        v3 = self.linear_(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
