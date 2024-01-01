
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, other=None):
        v1 = self.linear(x1)
        v2 = v1 + other if other is not None else v1
        v3 = F.relu(v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
