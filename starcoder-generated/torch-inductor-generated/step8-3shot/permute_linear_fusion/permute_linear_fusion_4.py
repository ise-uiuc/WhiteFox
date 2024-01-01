
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = v1.mul(2.0)
        v3 = v2.add(4.0)
        return v3
# Inputs to the model
x1 = torch.ones(1, 2, 2)
