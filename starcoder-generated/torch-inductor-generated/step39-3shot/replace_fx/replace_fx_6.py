
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v = torch.nn.functional.dropout(x)
        return torch.nn.functional.dropout(y)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
