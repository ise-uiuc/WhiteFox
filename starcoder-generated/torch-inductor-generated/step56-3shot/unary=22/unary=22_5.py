
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty((16, 32), requires_grad=True))
        self.b = torch.rand((16), requires_grad=True)
 
    def forward(self, x1):
        v1 = torch.matmul(self.w, x1) + self.b
        v2 = torch.tanh(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(16, 32)
