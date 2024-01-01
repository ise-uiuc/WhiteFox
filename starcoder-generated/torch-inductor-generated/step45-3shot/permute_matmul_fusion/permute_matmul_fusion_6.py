
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.matmul(x, x)
        x = x.permute(0, 2, 1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
