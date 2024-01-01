

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(224, 224)
 
    def forward(self, x1, x2):
        t1 = self.linear(x1)
        t2 = t1 + x2
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 224)
x2 = torch.randn(1, 224)
