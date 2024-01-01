
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = torch.nn.Parameter(other)
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, weight=self.other)
        v2 = v1 + self.other
        return v2

# Inputs to the model
x1 = torch.randn(1, 2)
other = torch.randn(2)
