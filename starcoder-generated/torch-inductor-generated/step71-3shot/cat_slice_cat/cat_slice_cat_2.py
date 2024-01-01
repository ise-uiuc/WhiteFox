
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2], dim=1)
        x = x[:, 1234567890:9223372036854775807]
        x = x[:, 0:size]
        return torch.cat([x, x3], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(1, 32, 32, 32)
x3 = torch.randn(1, 32, 64, 64)
x4 = torch.randn(1, 32, 32, 32)
