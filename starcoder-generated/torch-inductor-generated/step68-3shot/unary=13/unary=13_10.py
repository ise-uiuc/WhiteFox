
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(input_tensor)
        self.linear_1 = torch.nn.Linear(256, 64)
        self.linear_2 = torch.nn.Linear(64, 1)
 
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = torch.tanh(v1)
        v3 = v1 * v2
        return v3
# Initializing the model
__m__ = Model()

# Inputs to the model
x1 = torch.randn(1, 256)
