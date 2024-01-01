
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 32, 1, stride=1, bias=True)
        self.linear2 = torch.nn.Linear(16, 32, 1, stride=1, bias=True)
 
    def forward(self, x1, x2=None):
        t1 = self.linear1(x1)
        t2 = t1 + x2
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
