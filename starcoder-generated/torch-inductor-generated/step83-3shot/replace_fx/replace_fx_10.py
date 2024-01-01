
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.5)
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        x4 = torch.nn.functional.dropout(x3, p=0.5)
        return (x2, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
