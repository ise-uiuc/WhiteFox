
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(16, 8, bias=True)
        self.lin2 = torch.nn.Linear(8, 1, bias=True)
 
    def forward(self, x1):
        v1 = self.lin1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.lin2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
