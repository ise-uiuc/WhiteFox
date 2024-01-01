
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, w1, inp):
        v1 = w1 * inp
        v2 = v1 * w1
        return v1 + v2
# Inputs to the model
w1 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
