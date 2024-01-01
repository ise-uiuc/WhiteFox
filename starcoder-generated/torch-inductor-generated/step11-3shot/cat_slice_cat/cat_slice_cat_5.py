
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, x1, x2):
        v1 = torch.cat([x1, x2], dim=1)
        v2 = v1[:, 0:torch.iinfo(torch.int64).max]
        v3 = v2[:, 0:60]
        output1 = torch.cat([v1, v3], dim=1)
        return output1
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 186, 20, 20)
x2 = torch.randn(1, 178, 20, 20)
