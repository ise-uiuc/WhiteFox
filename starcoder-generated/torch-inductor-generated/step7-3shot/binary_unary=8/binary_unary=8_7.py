
class Model(torch.nn.Module):
    def __init__(self, input_tensor):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(torch.randn(1, 3, 64, 64))

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 1, 1, 1)
