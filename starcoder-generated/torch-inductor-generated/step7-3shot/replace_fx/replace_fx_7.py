
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, inplace=True)
        a2 = torch.nn.functional.dropout(x1, inplace=False)
        return a1, a2, a2
# Inputs to the model
x1 = torch.randn(1, 2)
