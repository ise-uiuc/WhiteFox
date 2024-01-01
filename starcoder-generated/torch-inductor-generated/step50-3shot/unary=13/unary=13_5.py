
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(64 * 64 * 3, 512)
 
    def forward(self, x1):
        f1 = x1.view((-1, 64 * 64 * 3))
        f2 = self.mlp(f1)
        g1 = torch.sigmoid(f2)
        g2 = g1 * f2
        return g2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
