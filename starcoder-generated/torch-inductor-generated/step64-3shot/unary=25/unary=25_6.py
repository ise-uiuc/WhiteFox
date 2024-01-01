
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1):
        v1 = F.linear(x1, self.linear.weight, self.linear.bias)
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
