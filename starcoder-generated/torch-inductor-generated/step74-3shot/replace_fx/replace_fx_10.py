
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.9)
    def forward(self, x):
        return F.dropout2d(x, p=0.9)
# Inputs to the model
x1 = torch.randn(10, 3, 7, 7)
