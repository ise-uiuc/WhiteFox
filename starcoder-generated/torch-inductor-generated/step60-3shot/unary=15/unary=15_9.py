
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.linear = torch.nn.Linear(256, 128)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.linear(v2.flatten(1))
        return v3
# Inputs to the model
x1 = torch.randn(4, 32, 224, 224)
