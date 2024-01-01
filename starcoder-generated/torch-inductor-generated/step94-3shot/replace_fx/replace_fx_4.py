
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.nn.functional.dropout(x1, p=0.2)
        x4 = torch.nn.functional.dropout(x2, p=0.2)
        x5 = torch.nn.functional.dropout(x3 + x4, p=0.2)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
