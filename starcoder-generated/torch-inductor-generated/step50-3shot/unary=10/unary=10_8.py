
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 2)
        self.linear2 = torch.nn.Linear(2, 2)
 
    def forward(self, x1):
        l1 = self.linear1(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        l6 = self.linear2(l5)
        return l6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
