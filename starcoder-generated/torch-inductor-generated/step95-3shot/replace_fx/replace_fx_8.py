
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, inplace=False)
        x3 = torch.nn.functional.dropout(x2, inplace=False)
        x4 = torch.nn.functional.dropout(x3, inplace=True)
        x5 = torch.nn.functional.dropout(x4, inplace=False)
        x6 = torch.nn.functional.dropout(x5, inplace=False)
        return (x6)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
