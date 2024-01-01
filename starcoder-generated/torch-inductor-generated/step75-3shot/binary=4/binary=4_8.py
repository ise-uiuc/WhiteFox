
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 9, bias=True)
    
    def forward(self, x1):
        t1 = self.linear(x1)
        t2 = t1 + x1
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(8)
