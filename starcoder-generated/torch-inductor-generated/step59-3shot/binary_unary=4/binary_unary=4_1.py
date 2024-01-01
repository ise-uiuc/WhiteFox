
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.rand(5, 5)
    
    def forward(self, x1, x2):
        v3 = relu(self.linear(x1) + x2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
