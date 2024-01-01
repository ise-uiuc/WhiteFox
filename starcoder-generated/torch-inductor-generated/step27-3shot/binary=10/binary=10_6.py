
class Model(torch.nn.Module):
    def __init__(self, other: torch.Tensor):
        super().__init__()
        self.l = torch.nn.Linear(3, 8)
        self.other = other
 
    def forward(self, x):
        v1 = self.l(x)
        v2 = v1 + self.other
        return v2

# Inputs to the model
x1 = torch.randn(1, 3)
