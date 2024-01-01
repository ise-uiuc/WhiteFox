
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a1 = torch.nn.functional.dropout(x1)
        a3 = 2 * a1
        a4 = torch.nn.functional.dropout(x2)
        a5 = a4 * a1
        return a5
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
