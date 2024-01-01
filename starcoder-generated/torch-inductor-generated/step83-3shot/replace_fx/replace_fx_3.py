
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        t1 = torch.rand_like(x1)
        x4 = torch.nn.functional.dropout(x3, p=0.5)
        return (x4, t1, x2)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
