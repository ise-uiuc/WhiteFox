
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.3)
        t2 = torch.rand_like(x1, requires_grad=True) + t1
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
