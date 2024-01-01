
class Model(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.linear.weight.data.copy_(weight)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = {'other': torch.randn(16)}
        v3 = torch.relu(v1 + v2['other'])
        return v3

# Initializing the model
weight = torch.randn(16, 1)
m = Model(weight)

# Inputs to the model
x1 = torch.randn(16)
