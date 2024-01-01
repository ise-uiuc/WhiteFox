
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1, x2, x3, x4):
        x5 = torch.nn.functional.dropout(x1)
        x6 = torch.nn.functional.dropout(x2)
        x7 = torch.nn.functional.dropout(x3)
        x8 = torch.nn.functional.dropout(x4)
        x9 = torch.nn.functional.dropout(self.dropout(x5))
        x10 = torch.nn.functional.dropout(self.dropout(x9))
        x11 = torch.nn.functional.dropout(self.dropout(x10))
        x12 = torch.nn.functional.dropout(self.dropout(x11))
        x13 = torch.nn.functional.dropout(self.dropout(x12))
        x14 = torch.nn.functional.dropout(self.dropout(x6))
        return x9, x13, x8
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
x3 = torch.randn(1, 2, 2)
x4 = torch.randn(1, 2, 2)
