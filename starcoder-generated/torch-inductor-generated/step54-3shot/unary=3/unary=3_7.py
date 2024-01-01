
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1 * 0.5
        v2 = x1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        return x1
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
