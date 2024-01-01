
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1 << 8, 1 << 8)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = F.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1_size = 1 << 8
x1 = torch.randn(1, x1_size)
