
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.rand_like(x, dtype=torch.double)
        s1 = F.dropout(x, p=0.5)
        return t1
# Inputs to the model
x = torch.randn(1, 2, 2)
