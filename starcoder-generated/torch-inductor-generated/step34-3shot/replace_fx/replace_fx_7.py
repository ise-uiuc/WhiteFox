
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = torch.nn.functional.dropout(x)
        y2 = torch.rand_like(x)
        y3 = torch.nn.functional.dropout(x, inplace=True)
        y4 = torch.rand_like(x)
        return y2
# Inputs to the model
x = torch.randn(1, 2, 2, 3)
