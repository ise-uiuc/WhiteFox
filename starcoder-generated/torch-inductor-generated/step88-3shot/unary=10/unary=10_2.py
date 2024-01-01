
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 10, bias=True)

    def forward(self, x0):
        # linear
        v1 = self.linear(x0)
        v2 = v1 + 3
        v3 = torch.nn.functional.hardtanh(v2, 0, 6)
        v4 = v3 / 6
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(2, 10)
