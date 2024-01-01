
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Initializing the `linear` layer with a `weight` and `bias`
linear = torch.nn.Linear(3, 8).double()
_x1 = torch.randn(1, 3).double()
_other = torch.randn(1, 8).double()

