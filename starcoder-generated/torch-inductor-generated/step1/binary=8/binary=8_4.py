
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, value):
        v1 = x + value
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
value = torch.randn(1, 3, 64, 64)
