
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(32, 100)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = F.threshold(v1, 0, 0) + 3
        v3 = v2 * 2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 32)
