
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, )
        t2 = torch.rand_like(x1)
        return t2
# Inputs to the model
x1 = torch.randn(2, 2, 2)
