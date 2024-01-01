
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, size):
        x2 = torch.cat([x1, x1], dim=1)
        v1 = x2[:, :, :size, :]
        v2 = x2[:, :, :, :size]
        v3 = torch.cat([v1, v2], dim=1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
size = 33
