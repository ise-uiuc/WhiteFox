
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 32)
        self.other = torch.nn.Parameter(torch.randn(32))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Inputs to the model
x1 = torch.randn(1, 3)
