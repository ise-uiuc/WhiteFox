
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 5, 5)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        if other is None:
            v2 = torch.nn.functional.relu(v1)
        else:
            v2 = v1 + other
        return v2

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 5, 1, 1)
