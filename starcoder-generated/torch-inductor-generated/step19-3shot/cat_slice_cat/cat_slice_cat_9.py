
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.cat([x[0], x[1], x[2]], 1)
        v2 = v1[:, :]
        v3 = v1[:, :]
        v4 = torch.cat([v1, v3], 1)
        return v4
 
# Initializing the model
m = Model()

# Inputs to the model
x = [torch.rand(32, 3, 224, 224), torch.rand(32, 3, 224, 224), torch.rand(32, 3, 224, 224)]
