
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout()
    def forward(self, x1):
        x2 = self.dropout(x1)
        x3 = torch.randn_like(x1)
        return torch.abs(x3) + x2
# Inputs to the model
x1 = torch.randn(10)
