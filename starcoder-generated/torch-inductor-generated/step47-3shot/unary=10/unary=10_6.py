
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(20, 2)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 + 3
        l2[l2 < 0] = 0
        l2[l2 > 6] = 6
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20)
