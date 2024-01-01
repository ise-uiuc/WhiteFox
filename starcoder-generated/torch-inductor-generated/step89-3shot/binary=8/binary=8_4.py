
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(1, 1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x, y, z):
        v1 = self.conv1(z)
        v2 = self.linear(v1) # <-- the operation triggering the pattern starts here
        v3 = v2 + x
        v4 = self.conv2(y)
        v5 = v3 + v4
        v6 = self.bn1(v5)
        v7 = v6 + v2 + v5 # <-- the operation triggering the pattern finally ends here
        return v7
# Inputs to the model
x = torch.randn(1, 3, 56, 56)
y = torch.randn(1, 3, 56, 56)
z = torch.randn(1, 3, 56, 56)
