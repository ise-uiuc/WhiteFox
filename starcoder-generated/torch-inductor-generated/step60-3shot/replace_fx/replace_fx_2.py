
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m2 = torch.nn.Linear(2, 2)
        self.b1 = torch.nn.BatchNorm1d(1, affine=False)
        self.dropout = torch.nn.functional.dropout
    def forward(self, x):
        self.b1.train()
        y1 = self.m2(x)
        y2 = self.dropout(y1, p=0.2, training=True)
        return y1 * y2, y2
# Inputs to the model
x1 = torch.randn(1, 2)
