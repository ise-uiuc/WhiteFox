
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
        self.dummy  = torch.nn.Parameter(torch.randn(1, )) # the other tensor parameter
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.dummy
        v3 = torch.relu(v2)
        return v3

# Initializing the model
__m__ = Model()

# Input to the model
x1 = torch.randn(1, 3)
