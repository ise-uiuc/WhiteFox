
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 7)
        self.linear2 = torch.nn.Linear(4, 7)
 
    def forward(self, x, other):
        v1 = self.linear1(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 4)
other = torch.randn(1, 7)
