
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 16)
 
    def forward(self, x1):
        return self.linear(x1) + 42 * torch.rand(1)

# Initializing the model
m = Model()
t_weight = m.linear.weight
t_bias = m.linear.bias

# Inputs to the model
x1 = torch.randn(1, 3, 4)
