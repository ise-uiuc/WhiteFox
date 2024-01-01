
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(20, 64)

    def forward(self, x2, x3):
        v1 = self.linear(x2)
        v2 = v1 + x3
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(10,20)
x3 = torch.randn(10,64)
