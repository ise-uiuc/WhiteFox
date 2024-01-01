
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(10, 10, bias=False)
 
    def forward(self, x1):
        v1 = self.l(x1)
        v2 = v1 + other 
        return v2

# Initializing the model and the extra tensor
m = Model()
other = torch.zeros(10)

# Inputs to the model
x1 = torch.randn(10)
