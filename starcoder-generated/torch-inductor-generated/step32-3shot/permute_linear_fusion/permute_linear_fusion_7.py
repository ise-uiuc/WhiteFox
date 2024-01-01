
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self, x1):
        return self.dropout(x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
