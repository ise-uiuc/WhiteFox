
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.nn.functional.dropout(x1)
        t2 = torch.nn.functional.dropout(x2)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
