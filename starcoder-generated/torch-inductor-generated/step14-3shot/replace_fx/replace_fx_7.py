
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        x2 = torch.sigmoid(x1)
        x3 = self.dropout(x1)
        return x2 * x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
