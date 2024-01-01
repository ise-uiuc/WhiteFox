
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:100] # 100 is any positive integer
        v4 = torch.cat([v1, v3], dim=1)
        return v4
     
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 64, 64)
x2 = torch.randn(1, 105, 64, 64)
