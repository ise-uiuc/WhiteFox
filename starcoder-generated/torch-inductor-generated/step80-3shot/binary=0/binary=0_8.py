
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x=torch.randn(4, 3, 224, 224), other=1):
        v1 = torch.ones_like(x)
        v2 = v1 + other
        x = None
        v3 = v2
        return v3
# Inputs to the model
