
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16, bias=True)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * 1
        l3 = 1 / 6 * l2
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
