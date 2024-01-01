
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(10, num_classes, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6

# Initializing the model
m = Model(1000)

# Inputs to the model
x1 = torch.randn(1, 10)
