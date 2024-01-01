
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.affine = torch.nn.Linear(16, 32, True)
 
    def forward(self, x1, other=None):
        v1 = self.affine(x1)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
