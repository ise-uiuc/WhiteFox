
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch._C._nn.dropout(x1, p=0.23, train=False, inplace=True)
        a2 = torch.rand_like(x1)
        return a2
# Inputs to the model
x1 = torch.randn(1, 2)
