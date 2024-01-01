
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        dim = 64
        input_dim = 64
        self.linear = torch.nn.Linear(input_dim, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64, 64)
