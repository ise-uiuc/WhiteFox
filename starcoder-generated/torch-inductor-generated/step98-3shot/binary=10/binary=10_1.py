
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(13, 4)
        self.linear2 = torch.nn.Linear(4, 5)
 
    def forward(self, x1, x2, other):
        v2 = self.linear1(x1) + other
        v3 = self.linear2(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 13)
x2 = torch.randn(1, 4)
other = torch.randn(1, 4)
