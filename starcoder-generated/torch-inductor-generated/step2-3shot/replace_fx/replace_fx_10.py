
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x1):
        c2 = x1 * 3
        return self.dropout(torch.nn.functional.dropout(c2))
# Inputs to the model
x1 = torch.randn(10)
