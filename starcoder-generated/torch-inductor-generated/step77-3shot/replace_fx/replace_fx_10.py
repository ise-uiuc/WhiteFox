
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x)
        return t1 + torch.rand_like(t1)
# Inputs to the model
x = torch.randn(1, 2, 2)
