
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4):
        x = torch.nn.functional.dropout(x1, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
