
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 50)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing inputs for model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 50)
