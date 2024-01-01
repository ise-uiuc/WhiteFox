
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=True)
        self.other = torch.rand(2, 3)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 3)
