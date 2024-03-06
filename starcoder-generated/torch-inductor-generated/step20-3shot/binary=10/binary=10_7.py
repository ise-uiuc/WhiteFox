
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(300, 100)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other!= None:
            v2 = v1 + other
        else:
            v2 = v1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 300)