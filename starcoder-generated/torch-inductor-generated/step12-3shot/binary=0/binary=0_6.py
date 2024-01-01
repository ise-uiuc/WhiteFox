
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(0)
    def forward(self, x1, other=True):
        v1 = self.dropout(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 256, 192, 178)
