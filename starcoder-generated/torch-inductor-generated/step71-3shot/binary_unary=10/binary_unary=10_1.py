
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_transform = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v1 = self.linear_transform(x1)
        v2 = v1 + x2
        v3 = v1 * 0.2
        v4 = torch.erf(v2 * 0.2)
        v5 = v3 * v4
        return v5

# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 8)
