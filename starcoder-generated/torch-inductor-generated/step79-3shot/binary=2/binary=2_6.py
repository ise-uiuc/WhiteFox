
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(8, 8)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 11.4
        v3 = self.linear(v2)
        v4 = v3 - 11.4
        return torch.argmax(v4)
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
