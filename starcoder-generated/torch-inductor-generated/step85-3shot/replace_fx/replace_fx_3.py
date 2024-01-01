
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if torch.rand(1) > 0.5:
            p = 1
            torch.nn.functional.dropout(x, p=1.0)
        else:
            p = 0
        x2 = torch.nn.functional.dropout(x, p=p)
        return x2
# Inputs to the model
x = torch.randn(8, 3)
