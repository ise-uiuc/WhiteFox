
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
        self.v1 = other
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.v1
        return v2

# Inputs to the model
other = torch.randn(1, 8)
x = torch.randn(1, 3)
