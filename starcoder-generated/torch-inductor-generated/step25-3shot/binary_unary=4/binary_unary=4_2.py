
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
 
    def forward(self, x1, other=torch.randn(1, 4).to(torch.float32), other_scalar=4.0):
        v1 = self.linear(x1)
        v2 = v1 + other
        v3 = relu(v2)
        v3_add = v3 + other_scalar
        return v4_add

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4).to(torch.float32)
