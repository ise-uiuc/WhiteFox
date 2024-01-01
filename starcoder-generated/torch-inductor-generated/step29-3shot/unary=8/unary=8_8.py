
class Model(torch.nn.Module):
    def __init__(self):
        self.conv_transpose = torch.rand([1, 64, 16, 16])
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sum(v1)
        torch.min(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
