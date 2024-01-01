
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2080, 3072)
 
    def forward(self, l1):
        t1 = self.linear(l1)
        t2 = t1 * torch.clamp(t1 + 3, max=6.0).to(t1.dtype)
        t3 = t2 / 6.0
        return t3

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 2080)
