
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64 * 64 * 3, 1) 
        self.other = torch.nn.Parameter(torch.randn(1))
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        v3 = relu(v2)
        return v3

# Initializing the model
m = Model()

# inputs to the model
x1 = torch.randn(8, 64, 64, 3)
