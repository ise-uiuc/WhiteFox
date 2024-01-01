
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.2)
    def forward(self, x):
        x = self.dropout(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
