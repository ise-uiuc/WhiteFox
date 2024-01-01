
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, c1):
        v1 = self.linear(x1)
        v2 = v1 - c1
        return v2

# Initializing and filling weights of the model
m = Model()
m.linear.weight.data.fill_(1.0)
m.linear.bias.data.fill_(2.0)

# Inputs to the model
c1 = torch.randn(1, 8)
x1 = torch.randn(1, 8)
