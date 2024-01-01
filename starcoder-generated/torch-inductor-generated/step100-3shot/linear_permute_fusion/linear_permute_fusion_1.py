
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm([1, 2, 2])
    def forward(self, x1):
        v1 = torch.nn.functional.layer_norm(x1, self.layernorm.weight, self.layernorm.bias, 1e-05, 1e-05)
        v2 = v1.reshape(2, 2)
        v1 = self.layernorm(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
