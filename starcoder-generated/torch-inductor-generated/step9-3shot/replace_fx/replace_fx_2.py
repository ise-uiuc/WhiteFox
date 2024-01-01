
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.nn.functional.dropout(x, p=0.4)
        x = x.add(2.0)
        x = x + y
        z = x.add(0.6)
        return z
# Inputs to the model
x = torch.randn(1, requires_grad=True)
