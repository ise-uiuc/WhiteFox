
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.gelu(x1)
        return v1.permute(0, 2, 1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
