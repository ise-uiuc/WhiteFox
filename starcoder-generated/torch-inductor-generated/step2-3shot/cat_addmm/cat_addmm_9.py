
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 8, bias=True)
 
    def forward(self, x1):
        v1 = self.l1(x1)
        v2 = torch.cat(v1, dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
