
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        v0 = x0
        v1 = v0[0].tolist()
        v2 = torch.tensor(v1)
        return v2
# Inputs to the model
x0 = torch.randn(3, 3, 3)
