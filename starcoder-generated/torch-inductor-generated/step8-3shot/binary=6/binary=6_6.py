
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()
# other is the tensor or scalar subtracted from every element
other = torch.randn(1, 1, 8)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
