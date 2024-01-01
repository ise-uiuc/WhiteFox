
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = other

    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - self.other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the model
other_value = 10
m = Model(other_value)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
