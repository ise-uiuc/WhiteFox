
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(...,...)
        v2 = x2.permute(...,...) # Transposed v2, which means v2[..., 0, 0] is v1[..., 0, 0].
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 4)
x2 = torch.randn(1, 2, 2, 4)
