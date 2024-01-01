
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v3 = torch.cat([x1, x2], dim=1)
        v5 = v3[:, 0:9223372036854775807]
        v4 = v5[:, 0:x1.size(2)]
        v7 = torch.cat([v3, v4], dim=1)
        return v7

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 125, 125)
x2 = torch.randn(1, 8, 125, 125)
