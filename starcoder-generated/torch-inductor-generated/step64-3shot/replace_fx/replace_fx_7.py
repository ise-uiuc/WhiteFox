
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0):
        x1 = F.dropout(x0, training=False)
        x2 = F.dropout(x1, p=0.5, training=False)
        x3 = F.dropout(x2, p=0.5)
        x4 = F.dropout(x3)
        return x4
# Inputs to the model
x0 = torch.randn(1, 2, 1, 1)
