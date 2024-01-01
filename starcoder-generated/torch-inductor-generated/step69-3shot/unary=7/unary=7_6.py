
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(32, 16, bias=True)
        self.linear2 = torch.nn.Linear(16, 8, bias=True)
 
    def forward(self, x1):
        l1 = self.linear1(x1)
        l2 = l1 * torch.clamp(torch.sum(l1) / 16, 0, 6)
        l3 = l2 / 6
        l4 = self.linear1(l3)
        l5 = l4 * torch.clamp(torch.sum(l4) / 16, 0, 6)
        l6 = l5 / 6
        return l6

# Intializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
