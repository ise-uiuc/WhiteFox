
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp=None):
        v1 = torch.mm(x1, x2)
        if inp is not None:
            v1 = v1 + inp

        v2 = torch.exp(v1)
        v4 = v2 + torch.tanh(v1) - v1
        v3 = torch.sigmoid(v2) + torch.clamp(v1, min=0.2)
        v5 = v1 + v2 * v3
        return v5
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
