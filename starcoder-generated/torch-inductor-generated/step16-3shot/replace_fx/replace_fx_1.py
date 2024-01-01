
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = torch.nn.functional.dropout(x)
        return t
# Inputs to the model
x1 = torch.randn(1, 2)
