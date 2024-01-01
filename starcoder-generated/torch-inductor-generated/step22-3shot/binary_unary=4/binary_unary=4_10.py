
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
 
    def forward(self, x1, o2):
        v1 = self.linear(x1)
        return v1 + o2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 3)
o2 = torch.randn(1, 3, 3)
