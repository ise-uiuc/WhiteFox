
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a0 = x1.narrow(0, 0, 1)
        a1 = torch.nn.functional.dropout(x1, p=0.23, training=a0.sum() > 0)
        a2 = torch.abs(a1)
        a3 = torch.nn.functional.dropout(a2, p=0.54, training=a0.sum() < 0)
        a4 = a2 * a3
        return a4
# Inputs to the model
x1 = torch.randn(8, 2)
