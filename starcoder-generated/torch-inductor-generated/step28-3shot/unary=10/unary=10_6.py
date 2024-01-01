
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l5 = torch.nn.Linear(10, 20, bias=False)
 
    def forward(self, x1):
        l1 = x1.mean(dim=[-1], keepdims=True)
        l2 = self.l5(l1)
        l3 = l2 + 3
        l4 = torch.clamp_min(l3, 0)
        l5 = torch.clamp_max(l4, 6)
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 10)
