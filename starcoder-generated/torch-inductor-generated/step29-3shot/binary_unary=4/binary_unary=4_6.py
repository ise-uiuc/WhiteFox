
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear_transform = Linear(8, 3)
 
    def forward(self, x1):
        v1 = self.linear_transform(x1)
        v2 = v1 + other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model(other=torch.randn(1, 3))

# Inputs to the model
x1 = torch.randn(1, 8)
