
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 32, bias=False)
        self.projOther = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, x1, x2=None):
        if(x2 is None):
            v1 = self.proj(x1)
            v2 = v1 + x1
        else:
            v1 = self.proj(x1)
            v2 = v1 + self.projOther(x2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
x2 = torch.randn(1, 8)
