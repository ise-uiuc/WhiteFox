
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.nn.functional.dropout(x1, p=0.5)
        x2 = torch.nn.functional.dropout(x1, p=0.2, inplace=True)
        x3 = torch.nn.functional.dropout(x1)
        x4 = torch.nn.functional.dropout(x1)
        return x1 + x2 + x3 + x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
