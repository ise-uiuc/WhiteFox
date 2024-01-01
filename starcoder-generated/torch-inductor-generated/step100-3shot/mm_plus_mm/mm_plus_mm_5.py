
class Model(torch.nn.Module):
    def forward(self, h, i, g, z):
        T0 = g * i
        T1 = z * g
        T2 = i * z
        T3 = g * g
        T4 = z * g
        T5 = z * z
        T6 = g * g
        T7 = i * i
        out = T0
        return out
# Inputs to the model
h = torch.randn(2, 2)
i = torch.randn(2, 2)
g = torch.randn(2, 2)
z = torch.randn(2, 2)
