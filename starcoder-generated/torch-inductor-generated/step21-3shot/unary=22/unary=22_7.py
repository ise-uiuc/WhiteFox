
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Printing initial weights
print('Initial linear weight:', m.linear.weight)
print('Initial linear bias:', m.linear.bias)

# Inputs to the model
x1 = torch.randn(1, 64)
