
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.nn.functional.dropout(x, p=0.5, out=y)
        return x + 2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
y1 = torch.randn(1, 2, 2)
