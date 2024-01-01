
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.4)
    def forward(self, x):
        y1 = self.dropout(x)
        y2 = y1 ** 1
        z = torch.rand_like(y1)
        y3 = y2 + z
        return y3
# Inputs to the model
x = torch.randn(1, 2)
