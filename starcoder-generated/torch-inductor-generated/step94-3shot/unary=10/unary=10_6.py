
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8, bias=False)
        self.add = torch.nn.quantized.FloatFunctional()
        self.clamp = torch.nn.Clip()

    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = self.add.add(l1, 3)
        l3 = self.clamp.relu6(l2)
        l4 = l3 / 6
        return l4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
