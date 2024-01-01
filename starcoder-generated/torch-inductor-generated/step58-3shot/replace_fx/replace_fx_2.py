
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = F.dropout(x1, p=0.5, training=True)
        x4 = x3 + x2
        x5 = F.dropout(x3, p=0.5, training=False)
        x6 = F.dropout(x4, p=0.5, training=True)
        x7 = torch.rand_like(x4)
        x8 = torch.rand_like(x1)
        return (x5,)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
