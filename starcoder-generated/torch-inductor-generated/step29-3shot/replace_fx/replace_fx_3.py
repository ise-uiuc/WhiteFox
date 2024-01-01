
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.nn.functional.dropout(x1, p=0.1, inplace=True)
        w2 = torch.rand_like(x1)
        t1 = y1 + w2
        return t1
# Inputs to the model
x1 = torch.randn(1, 3, 3)
