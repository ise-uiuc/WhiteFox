
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=True)
 
    def forward(self, x1, x2=torch.ones(3, dtype=torch.float)):
        v1 = self.linear(x1)
        return v1 + x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2)
