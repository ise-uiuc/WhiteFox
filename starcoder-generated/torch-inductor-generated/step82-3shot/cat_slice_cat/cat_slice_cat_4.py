
class Model(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:9223372036854775807]
        v3 = v2[:, 0:2725]
        v4 = torch.cat([v1, v3], dim=1)
        return []

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2725, 112, 112)
x2 = torch.randn(1, 2725, 112, 112)
