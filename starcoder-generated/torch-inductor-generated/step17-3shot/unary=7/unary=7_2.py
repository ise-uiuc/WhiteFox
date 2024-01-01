
def selu(x):
    return 1.0507 * torch.nn.functional.leaky_relu(x, 0.0167, inplace=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * torch.clamp(torch.nn.functional.linear(x1, self.linear.weight, bias=self.linear.bias), min=0, max=6)
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
