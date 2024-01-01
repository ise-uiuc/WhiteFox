
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        return v2

# Initializing the model
m2 = Model()

# Inputs to the model
x2 = torch.randn(1, 3) 
x1 = torch.randn(1, 3, 64, 64)
