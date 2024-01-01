
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.0, max_value=+1.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
 
    def forward(self, x):
        v1 = torch.clamp(x, min=self.min_value)
        v2 = torch.clamp(self.conv(x), min=self.min_value)
        return torch.clamp(v1 + v2, max=self.max_value)

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
