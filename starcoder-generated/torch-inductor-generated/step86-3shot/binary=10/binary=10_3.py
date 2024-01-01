
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Linear(2, 8)
 
    def forward(self, x):
        v1 = self.op1(x)
        return v1 + self.op1.weight

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
