
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(65536, 65536, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(65536, dtype=torch.float32), requires_grad=True)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.add(v1, other=other)
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model(other)

# Inputs to the model
x1 = torch.randn(128, 65536)
