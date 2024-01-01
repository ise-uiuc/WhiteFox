
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b1 = F.dropout(x1, p=0.4, training=True)
        b2 = F.dropout(x1, p=1.0, training=False)
        b3 = F.dropout(x1, p=0.0)
        b4 = F.dropout(x1, training=True)
        b5 = F.dropout(x1, training=False)
        return (b1, b2, b3, b4, b5)
# Inputs to the model
x1 = torch.randn(1, 2, 3, 4)
