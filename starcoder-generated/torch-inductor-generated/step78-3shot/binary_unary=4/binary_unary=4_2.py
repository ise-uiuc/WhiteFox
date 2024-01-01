
class Model(torch.nn.Module):
    def __init__(self, other_tensor):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return relu(v2)

# Initializing the model
m = Model(torch.randn(10, 10))

# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
