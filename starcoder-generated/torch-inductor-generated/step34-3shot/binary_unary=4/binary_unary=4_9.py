
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x=x1, weight=None, bias=None)
        v2 = v1 + other
        v3 = F.relu(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
