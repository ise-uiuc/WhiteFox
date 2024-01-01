
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = torch.slice(v1, 0, 9223372036854775807, 1, 1, 0, 9223372036854775807)
        v3 = torch.slice(v2, 0, size, 1, size)
        v4 = torch.cat([v1, v3], dim=1)
        v5 = v5 + x2
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
size = x2.shape[-(-'b')]
