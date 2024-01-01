
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 16, bias=False)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()
init_weight = torch.randn(16, 128)
m.linear.weight.data = init_weight
m.other = torch.rand(16, )

# Inputs to the model
x1 = torch.randn(128, )
