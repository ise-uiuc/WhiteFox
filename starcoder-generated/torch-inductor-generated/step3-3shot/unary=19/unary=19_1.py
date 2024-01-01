
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear10 = torch.nn.Linear(10, 10, bias=False)
        self.linear11 = torch.nn.Linear(10, 10, bias=False)
 
    def forward(self, x0):
        v1 = self.linear10(x0)
        v2 = torch.sigmoid(v1)
        v3 = self.linear11(v2)
        v4 = torch.sigmoid(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(1, 10)
