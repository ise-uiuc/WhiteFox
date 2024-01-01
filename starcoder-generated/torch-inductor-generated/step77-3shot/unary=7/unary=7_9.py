
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, l1):
        l2 = l1 * torch.clamp(self.linear(l1) + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 8)
