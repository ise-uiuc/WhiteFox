
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bmm = torch.Tensor.bmm
    def forward(self, x1, x2):
        v0 = x1.permute(0, 2, 1)
        v1 = self.bmm(v0, x2)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
