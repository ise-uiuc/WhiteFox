
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        u1 = torch.nn.functional.dropout(x1, p)
        u2 = torch.nn.functional.dropout(x1, p=0.)
        return u1 - u2
# Inputs to the model
x1 = torch.randn(1, 2, 3, 5)
