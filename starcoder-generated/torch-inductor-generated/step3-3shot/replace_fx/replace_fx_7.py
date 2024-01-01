
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x1, x2, x3):
        y1 = x2 + x3
        y2 = self.dropout(y1)
        return self.dropout(x1), y2
# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 500000)
x3 = torch.randn(1, 5000)

