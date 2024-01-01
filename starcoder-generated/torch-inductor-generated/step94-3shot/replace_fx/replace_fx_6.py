
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.2)
        x3 = torch.nn.functional.dropout(x2, p=0.2)
        x4 = torch.nn.functional.dropout(self.dropout(x3), p=0.2)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
