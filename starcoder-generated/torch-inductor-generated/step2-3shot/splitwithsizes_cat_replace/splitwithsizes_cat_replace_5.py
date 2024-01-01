
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
 
    def forward(self, x1):
        out = self.softmax(x1)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1000, 1000)
torch.split(x1, 100, dim=1)
torch.cat([x2[:, j] for x2 in v], dim=1)
