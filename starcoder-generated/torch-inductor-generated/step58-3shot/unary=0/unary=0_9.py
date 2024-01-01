
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3, stride=3, padding=0)  # type: ignore[arg-type]
    def forward(self, x2):
        v8 = 0.5
        v4 = x2 - 0.2167882471220062
        v1 = self.conv(v4)
        v1.requires_grad_(True)
        v2 = v1 * v8
        v3 = v1 * v1
        v5 = v3 * v1
        v6 = v5 * 0.044715
        v7 = v1 + v6
        v9 = v7 * 0.7978845608028654
        v10 = torch.tanh(v9)
        v11 = v10 + 1
        v12 = v2 * v11
        return v12
# Inputs to the model
x2 = torch.randn((1, 1, 32, 32))
