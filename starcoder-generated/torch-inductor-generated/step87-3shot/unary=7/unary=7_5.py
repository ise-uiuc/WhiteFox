
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 6)
 
    def forward(self, x2):
        l1 = self.linear(x2)
        l2 = torch.clamp(self.linear(x2) + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 10)
