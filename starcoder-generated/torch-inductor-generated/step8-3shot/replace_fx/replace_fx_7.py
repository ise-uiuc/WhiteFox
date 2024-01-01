
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.4)
        a2 = torch.nn.functional.dropout(a1, p=0.2)
        a3 = torch.nn.functional.dropout(a2, p=0.3)
        a4 = torch.nn.functional.dropout(a3, p=0.1)
        a5 = a1 - a4
        a6 = torch.nn.functional.dropout(a5, p=0.2)
        return a6 * a6
# Inputs to the model
x1 = torch.randn(4, 4)
