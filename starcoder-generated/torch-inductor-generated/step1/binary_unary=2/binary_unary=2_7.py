
class Model(torch.nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.value = torch.nn.Parameter(init_value)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.value.view(1, 1, 1, 1)
        v3 = v1 - v2
        v4 = torch.relu(v3)
        return v4

# Initializing the model
initial_value = torch.randn(1, 3, 64, 64)
m = Model(torch.nn.Parameter(initial_value))

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
