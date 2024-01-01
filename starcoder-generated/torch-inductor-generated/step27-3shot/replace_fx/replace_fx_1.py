
class model(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.dropout_1 = torch.nn.Dropout(p=self.p)
        self.dropout_2 = torch.nn.Dropout(p=self.p)
        self.dropout_3 = torch.nn.Dropout(p=self.p)
    def forward(self, x):
        v1 = self.dropout_1(1)
        v2 = self.dropout_2(2)
        v3 = self.dropout_3(3)
        out = x + v1 + v2 + v3
        return out
# Inputs to the model
x1 = torch.randn(1, 2, 2)
