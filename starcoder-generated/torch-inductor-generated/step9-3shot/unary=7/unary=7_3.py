
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
 
    def forward(self, input):
        l1 = self.linear(input)
        l2 = l1 * torch.clamp(l1 + 3, min=0, max=6)
        l3 = l2 / const(6)
        return l3


# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(2, 1)
