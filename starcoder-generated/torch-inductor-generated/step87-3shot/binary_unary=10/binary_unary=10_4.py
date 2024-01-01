
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 8)
 
    def forward(self, __input_name__):
        v1 = self.linear(x1)
        v2 = v1 + 1
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 64)
x2 = torch.randn(2, 8)
x2 = torch.randn(2, 8)
