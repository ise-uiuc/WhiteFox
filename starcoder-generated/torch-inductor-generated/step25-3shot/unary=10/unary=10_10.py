
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 3)
 
    def forward(self, x1):
        l1 = self.fc1(x1)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        return l4 / 6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(3, 2)
