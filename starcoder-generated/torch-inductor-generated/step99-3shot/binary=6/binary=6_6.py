
class Model(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.tensor(weight, requires_grad=True)

    def forward(self, x1):
        v1 = torch.addmm( bias=None, input=x1, weight=self.weight)
        v2 = v1 - 0.005
        return v2

# Initializing the model
m = Model(torch.zeros(6, 3))

# Inputs to the model
x1 = torch.randn(1, 6)
