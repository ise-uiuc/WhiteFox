
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, bias=None):
        x2 = torch.nn.functional.dropout(x1)
        if bias is not None:
            x2 = x2 + bias
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, bias=None):
        x2 = torch.nn.functional.dropout(x1, p=0.2)
        if bias is not None:
            x2 = x2 + bias
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
bias = torch.randn(2)
