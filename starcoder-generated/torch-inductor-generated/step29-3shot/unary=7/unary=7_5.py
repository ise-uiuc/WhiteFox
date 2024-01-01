
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
 
    def forward(self, x1):
        t1 = self.linear1(x1)
        t2 = t1 * torch.clamp(t1 + 3, min=0, max=6)
        t3 = t2 / 6
        return t3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)

