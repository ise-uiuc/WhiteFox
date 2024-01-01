
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        x4 = F.dropout(x1, p=0.5, training=True)
        x5 = torch.rand_like(x1)
        x6 = F.dropout(x1, p=0.5)
        x7 = F.dropout(x1, p=0.5, training=True)
        return x7
# Inputs to the model
x1 = torch.randn(2, 2, 2)
