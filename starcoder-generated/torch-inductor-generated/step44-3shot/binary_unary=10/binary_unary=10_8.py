
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12, 24, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(1, 24, dtype=torch.float32))
 
    def forward(self, x1):
        w1 = self.linear.weight.to(x1.device)
        v2 = torch.matmul(x1, w1)
        v3 = v2 + self.bias
        v4 = torch.nn.functional.relu(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 12, dtype=torch.float32)
