
class Model(torch.nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4, False)
        self.linear.weight.data = weight
        self.linear.bias.data = bias
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - other
        return v2

# Initializing the model
m = Model(torch.randn(4, 3), torch.randn(4))

# Inputs to the model
x1 = torch.randn(1, 3)
other = torch.randn(1, 4)
