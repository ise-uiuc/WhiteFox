
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28, 30)

    def forward(self, x, other):
        v1 = self.linear(x)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model()

# Inputs to the model. Setting the correct size of the tensor for `other` will have different behavior in the `torch.ops.aten._sub` operator
x = torch.randn(1, 3, 3)
other = torch.randn(1, 4, 5)
