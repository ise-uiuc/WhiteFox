
class Model(torch.nn.Module):
    def __init__(self, otherA=1, otherB=2, otherC=3):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
 
    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model(otherA=1, otherB=2, otherC=3)

# Inputs to the model
x = torch.randn(1, 8)
other = torch.tensor([4, 5, 6, 7])
