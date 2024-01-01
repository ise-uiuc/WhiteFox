
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Linear(8, 4)
        self.c = torch.nn.Linear(8, 4)
 
    def forward(self, x1):
        r1 = self.m(x1)
        r2 = self.c(x1)
        return (r1 * (r2 + 3).clamp(min=0, max=6)) / 6

# Initializing the model
m = Model()

# Inputs to the model
x4 = torch.randn(1, 8, 32, 32)
