
class Model(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.value = value
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 + self.value
        return torch.relu(v2)

# Initializing the model
m = Model(1)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
m2 = Model(m)
m2.eval()
