
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.nn.Transformer()
    def forward(self, v1):
        v2 = self.t(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 32, 64)
