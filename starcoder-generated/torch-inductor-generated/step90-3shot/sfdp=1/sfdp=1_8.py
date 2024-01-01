
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)
 
    def forward(self, x1, x2, x3):
        q = self.proj(x1)
        k = x2
        k2 = torch.nn.functional.softmax(torch.sum(x3, dim=1), dim=1)
        return torch.cat([q, k, k2], dim=1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 1, 4)
x2 = torch.randn(4, 5, 3)
x3 = torch.randn(3, 6, 4)
