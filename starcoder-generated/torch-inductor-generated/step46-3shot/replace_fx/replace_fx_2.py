
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout2d(x1, p=0.5)
        x3 = F.dropout2d(x2, p=0.5, training=True)
        x4 = F.dropout2d(x1, p=0.5, training=False)
        x5 = F.dropout2d(x2, p=0.5)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
