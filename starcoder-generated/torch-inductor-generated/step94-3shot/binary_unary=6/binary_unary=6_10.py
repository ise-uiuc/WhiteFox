
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 8)

    def forward(self, x1, x2):
        v1 = torch.addmm(x1, self.linear.weight, self.linear.bias)
        v2 = torch.sub(v1, x2)
        return nn.ReLU()(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 6)
