
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, add_tensor):
        v1 = self.conv(x1)
        v2 = v1 + add_tensor
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
add_tensor = torch.randn(1, 8, 64, 64)
