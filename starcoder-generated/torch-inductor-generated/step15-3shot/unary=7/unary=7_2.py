
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 512, bias=False)
        self.min_val = torch.nn.Parameter(torch.tensor(0.))
        self.max_val = torch.nn.Parameter(torch.tensor(6.))
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(l1 + self.min_val, self.min_val, self.max_val)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
