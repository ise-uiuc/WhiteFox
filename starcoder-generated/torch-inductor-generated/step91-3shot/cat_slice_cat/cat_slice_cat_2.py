
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(65535, 65535, bias=False)
        self.s = torch.nn.Linear(65535, 65535, bias=True)
 
    def forward(self, x1, x2):
        o1 = self.m(x1)
        o2 = o1 + self.s(x2)[:, 0:65535]
        return o2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 65535)
x2 = torch.randn(1, 65535)
