
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1,bias=True)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1 + x
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
