
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other # 'other' is a constant tensor/scalar
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(8)
