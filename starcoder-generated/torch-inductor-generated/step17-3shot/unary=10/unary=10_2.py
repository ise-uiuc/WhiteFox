
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(3, 8, bias=True)
 
    def forward(self, x2):
        l1 = self.linear(x2)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3)
