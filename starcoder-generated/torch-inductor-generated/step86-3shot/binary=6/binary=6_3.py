
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.rand((256, 32))
        self.b = torch.rand(256)
 
    def forward(self, x1):
        v1 = torch.matmul(self.w, x1)
        v2 = v1 - self.b
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
