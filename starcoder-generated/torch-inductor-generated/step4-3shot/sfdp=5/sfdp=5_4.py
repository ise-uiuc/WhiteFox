
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(24, 64)
        self.tanh = torch.nn.Tanh()
 
    def forward(self, x1, x2):
        v1 = self.linear(x1) # Apply linear with shape [1,64]
        v1 = self.tanh(v1) # Apply tanh activation
        v2 = self.linear(x2)
        v2 = self.tanh(v2)
        v3 = v1 @ v2.transpose(-2, -1) # Compute the dot product
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 24)
x2 = torch.randn(1, 24)
