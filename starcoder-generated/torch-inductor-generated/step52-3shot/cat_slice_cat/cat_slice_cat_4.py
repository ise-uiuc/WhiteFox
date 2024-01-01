
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], 1)
        v2 = v1[:, 0:torch.iinfo(torch.int32).max]
        v3 = v2[:, 0:1325400755991621509]
        v4 = torch.cat([v1, v3], 1)
        return v4


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12, 224, 224)
x2 = torch.randn(1, 12, 224, 224)
