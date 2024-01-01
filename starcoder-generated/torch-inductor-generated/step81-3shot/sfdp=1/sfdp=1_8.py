
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm([2, 4], elementwise_affine=False)
 
    def forward(self, x1):
        v1 = self.layernorm(x1)
        v2 = torch.matmul(v1, v1.transpose(-2, -1))
        v3 = v2 / 16
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(5, 2, 4)
