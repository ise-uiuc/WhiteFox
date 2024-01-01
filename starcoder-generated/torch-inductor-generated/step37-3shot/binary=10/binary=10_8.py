
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(20, 5)
        self.linear2 = torch.nn.Linear(5, 64)
 
    def forward(self, x1, other=None):
        v1 = self.linear1(x1)
        v2 = v1 + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
other = torch.randn(1, 5)
