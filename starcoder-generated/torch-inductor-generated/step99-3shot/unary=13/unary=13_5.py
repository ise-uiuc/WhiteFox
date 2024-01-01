
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        input_tensor = torch.randn(10, 20)
        weight = torch.randn(20, 10)
        bias = torch.randn(10, )
        self.linear = torch.nn.Linear(20, 10, bias=True)
        self.linear.weight = torch.nn.Parameter(weight)
        self.linear.bias = torch.nn.Parameter(bias)
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.sigmoid()
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(50, 20)
