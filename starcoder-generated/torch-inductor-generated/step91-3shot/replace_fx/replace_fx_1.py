
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout1 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
        self.dropout2 = torch.nn.ModuleList([torch.nn.Dropout(p=0.5) for _ in range(10)])
    def forward(self, x1):
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        x4 = F.dropout(x1, p=0.5)
        x5 = torch.nn.functional.dropout(x1)
        x6 = torch.nn.functional.dropout(x1, p=0.5, inplace=True)
        return x3 + x4 + x5 + x6
# Inputs to the model
x1 = torch.randn(1, 2, 2)
