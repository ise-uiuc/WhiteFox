
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.other = torch.randn(1, 8, 64, 64)
 
    def forward(self, x1):
        c = self.conv(x1)
        c1 = torch.add(c, self.other)
        c2 = torch.relu(c1)
        return c2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
