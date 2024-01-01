
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b1 = torch.nn.functional.dropout(x, p=0.3, inplace=True)
        a1 = b1.flatten(0, 1)
        p1 = torch.nn.functional.dropout(a1, p=0.4, inplace=False)
        return a1
# Inputs to the model
x1 = torch.randn(20, 30, 40)
