
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x):
        t = self.dropout(x)
        return t
# Inputs to the model
x1 = torch.randn(1, 3, 3)
