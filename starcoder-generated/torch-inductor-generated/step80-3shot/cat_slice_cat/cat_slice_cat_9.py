
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:x2.size()]
        v4 = torch.cat([v1, v3], dim=1)
        return [torch.relu(v1) + v4, v1]

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 77, 77)
x2 = torch.randn(1, 128, 77, 77)
