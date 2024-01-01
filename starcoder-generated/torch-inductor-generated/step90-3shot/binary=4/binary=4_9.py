
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=8, out_features=8, bias=False)
        self.bias = torch.nn.Parameter(torch.arange(0, 8), requires_grad=True)
 
    def forward(self, x1, y1):
        v1 = self.linear(x1)
        v2 = v1 + y1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
y1 = torch.randn(1, 8)
