
class Model(torch.nn.Module):
    def __init__(self, x1):
        super().__init__()
        self.features = torch.nn.Linear(3, 10)
        self.conv = torch.nn.Conv2d(10, 16, 3, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        return v3

# Initializing the model
m = Model(x1)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
