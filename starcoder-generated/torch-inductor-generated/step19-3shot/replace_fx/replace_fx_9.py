
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d()
    def forward(self, x):
        out1 = torch.nn.functional.dropout(x, p=0.2)
        out2 = self.dropout(x)
        out = out1 + out2
        return out
# Inputs to the model
x = torch.randn(32, 16, 30, 50)
