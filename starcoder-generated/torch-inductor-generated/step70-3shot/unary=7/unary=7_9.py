
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6)
 
    def clamp_fn(self, min_p, max_p):
        def cl(x):
            return torch.clamp(x, min=min_p, max=max_p)
        return torch.nn.ModuleDict({'fn': cl})
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * self.clamp_fn(0, 6)(l1 + 3.0)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
