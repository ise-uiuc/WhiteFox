
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
 
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = v2 + other
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 299, 299)
