
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1):
        v1 = self.batch_norm(x1)
        v2 = torch.cat(x1, v1)
        v3 = v2[:, 0:9223372036854775807]
        v4 = v3[:, 0:v3.size(-1)]
        v5 = torch.cat([v2, v4], dim=1)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
