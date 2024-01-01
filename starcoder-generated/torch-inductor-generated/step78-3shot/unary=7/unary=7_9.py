
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
        torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * v1.clamp(0, 6) + 3
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
