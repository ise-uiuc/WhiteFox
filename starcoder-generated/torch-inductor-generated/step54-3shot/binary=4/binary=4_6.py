
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(4, eps=0)
    
    def forward(self, x1, other=None):
        v1 = self.norm(x1)
        if other is not None:
            c = torch.nn.functional.linear(v1, other)
        else:
            c = v1
        return c
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(4, 4)
