
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 5)
        self.l2 = torch.nn.Linear(3, 7)
 
    def forward(self, x1, x2):
        v1 = self.l1(x1)
        v2 = v1 + x2
        v3 = self.l2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(6, 3)
x2 = torch.randn(6, 3)
out = m(x1, x2)

