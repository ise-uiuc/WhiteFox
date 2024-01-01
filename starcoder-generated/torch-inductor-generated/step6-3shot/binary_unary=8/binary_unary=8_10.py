
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(8, 8)
        self.linear2 = torch.nn.Linear(8, 1)
 
    def forward(self, x1, x2):
        v1 = self.conv(x1) + self.linear(v1)
        v3 = torch.relu(v2)
        return v4

# Initializing the model
model = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 1, 1)
