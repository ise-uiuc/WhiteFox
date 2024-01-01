
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, other=1, padding1=2, padding3 = None):
        if padding3 == None:
            padding3 = torch.randn(1, 3, 64, 64)
        v1 = torch.randn(1, 3, 64, 64)
        if padding1 == 2:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
