
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(197, 100)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1.view(-1, 197)).clamp(0, 6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
