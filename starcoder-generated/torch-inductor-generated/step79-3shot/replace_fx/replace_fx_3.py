
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout0 = torch.nn.Dropout(0.90)
        self.dropout1 = torch.nn.Dropout(0.80)
    def forward(self, x1):
        x1 = self.dropout0(x1)
        x2 = torch.rand_like(x1)
        x3 = self.dropout1(x2)
        return x2, x3
# Inputs to the model
x1 = torch.randn(16, 256, 8)
