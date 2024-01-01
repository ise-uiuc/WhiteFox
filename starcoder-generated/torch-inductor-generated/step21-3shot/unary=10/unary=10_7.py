
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 8)
        self.linear2 = torch.nn.Linear(8, 4)
        self.linear3 = torch.nn.Linear(4, 2)
 
    def forward(self, x1):
        l1 = torch.tanh(self.linear1(x1))
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        v1 = torch.tanh(self.linear2(l5))
        v2 = v1 + 2
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v4 / 6
        v6 = torch.tanh(self.linear3(v5))
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
