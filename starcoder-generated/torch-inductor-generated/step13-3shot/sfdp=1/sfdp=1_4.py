
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / math.sqrt(8)
 
    def forward(self, q, k, v):
        return torch.matmul(torch.matmul(q, k.transpose(-2, -1)), v).div(self.scale_factor)

# Initializing the model
m = Model()

# Inputs to the model
q = torch.randn(1, 3, 64, 64)
k = torch.randn(1, 3, 64, 64)
v = torch.randn(1, 3, 64, 64)
