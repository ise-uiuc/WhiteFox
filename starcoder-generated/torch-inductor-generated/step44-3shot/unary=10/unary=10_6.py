
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        f1 = self.linear(x1)
        f2 = f1 + 3
        f3 = torch.clamp(f2, min=0)
        f4 = torch.clamp(f3, max=6)
        f5 = f4 / 6
        return f5

# Initializing model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
