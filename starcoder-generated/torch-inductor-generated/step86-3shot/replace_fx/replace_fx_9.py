
class Model(torch.nn.Module):
    def __init__(self):
        self.dropout = torch.nn.Dropout()
    def forward(self, x):
        x1 = self.dropout(x)
        x2 = self.dropout(x)
        x3 = x1 + x2
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
