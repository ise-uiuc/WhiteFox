
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b1 = torch.nn.functional.dropout(x)
        b2 = torch.nn.functional.dropout(x, p=0.4)
        b3 = torch.nn.functional.dropout(x, p=0.5)
        b4 = torch.nn.functional.dropout(x, p=0.2)
        b5 = torch.nn.functional.dropout(x, inplace=True)
        b6 = torch.nn.functional.dropout(x, p=0.4, inplace=True)
        b7 = torch.nn.functional.dropout(x, p=0.5, inplace=True)
        b8 = torch.nn.functional.dropout(x, p=0.2, inplace=True)
        return 1
# Inputs to the model
x = torch.randn(1)
