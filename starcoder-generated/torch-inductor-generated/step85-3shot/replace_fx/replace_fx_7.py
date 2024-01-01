
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = torch.nn.functional.dropout(x, p=0.5)
        x3 = x2 + x2
        return x3 + x2
# Inputs to the model
x = torch.randn(8, 3)
