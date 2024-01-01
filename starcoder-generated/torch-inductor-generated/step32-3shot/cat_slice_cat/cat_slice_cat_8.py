
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v3 = torch.cat([x1, x2], dim=1)
        v1 = v3[:, 0:9223372036854775807]
        v2 = v1[:, 0:10]
        result = torch.cat([v3, v2], dim=1)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 60)
x2 = torch.randn(1, 9, 60)
