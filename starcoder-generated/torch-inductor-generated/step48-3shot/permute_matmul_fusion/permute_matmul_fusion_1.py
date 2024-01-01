
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        out1 = torch.matmul(x1, x2)
        out2 = torch.matmul(x2, x1)
        return (out1, out2)
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
