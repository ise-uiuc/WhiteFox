
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(p=0.)
    def forward(self, x):
        h = self.linear(x)
        z = self.dropout(h)
        return z
# Inputs to the model
x = torch.randn(1, 3)
