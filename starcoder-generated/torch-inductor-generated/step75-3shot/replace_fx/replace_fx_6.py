
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        (x2, x3, _, x4, t) = (x1, x1, F.dropout(x1, p=0.5), F.dropout(x1, p=0.5), torch.rand_like(x1))
        return (x2, x4, t)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
