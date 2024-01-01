
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 + 3
        l3 = l2.clamp(0, 6)
        l4 = l3 / 6
        return l4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
