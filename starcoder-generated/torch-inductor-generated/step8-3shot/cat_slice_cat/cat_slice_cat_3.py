
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_size = 100000000
 
    def forward(self, x1, x2):
        v3 = []
        v3.append(x1)
        v3.append(x2)
        v1 = torch.cat(v3, dim=1)
        v2 = v1[:, 0:self.max_size]
        v2[:, 0:self.max_size]
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 500, 1, 1)
x2 = torch.randn(1, 300, 1, 1)
