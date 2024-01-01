
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=False)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.hardtanh(v1, 0, 6)
        v3 = v2 + 3
        v5 = v3 / 6
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 =  torch.randn(16, 4)
