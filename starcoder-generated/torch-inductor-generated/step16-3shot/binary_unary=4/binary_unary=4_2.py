
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1, other):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = torch.nn.functional.relu(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 64)
__other__ = torch.randn(1, 64)
