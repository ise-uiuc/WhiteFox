
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)
 
    def forward(self, x1):
        self.linear.weight.data *= 0.00006
        self.linear.bias.data *= 0.00006
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(l1, min=0, max=6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
