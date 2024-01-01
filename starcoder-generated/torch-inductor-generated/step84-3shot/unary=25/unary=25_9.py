
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = torch.nn.Parameter(v2.shape, requires_grad=False)
        v4 = v1 * 0.2
        torch.where(v2, v1, v4, out=v3)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
