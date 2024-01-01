
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.nn.Parameter(torch.randn(10))
    def forward(self, x2):
        x1 = self.x1
        x1 = x1 + x2
        x2 = torch.nn.functional.dropout(x1, 0.8)
        x1 = torch.nn.functional.dropout(x2, 0.7)
        return x1, x2
# Inputs to the model
x1 = torch.randn(1, 10)
x2 = torch.randn(1, 10)
