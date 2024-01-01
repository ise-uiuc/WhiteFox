
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1 + 2
        x3 = torch.nn.functional.dropout(x2)
        x4 = x2 + 3
        x5 = torch.tanh(x4)
        return x5
# Inputs to the model
x1 = torch.randn(3, 3)
