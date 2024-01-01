
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 16, bias=True)

    def forward(self, x1):
        i1=x1.shape[0]
        l1 = self.linear(x1)
        l2 = l1 * F.hardtanh(l1 + 3, 0, 6)
        l3 = l2 / 6
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
