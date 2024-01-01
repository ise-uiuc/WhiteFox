
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = F.dropout(x1, p=0.5, training=True)
        x4 = F.dropout(x1, p=0.5, training=False)
        z1 = torch.rand_like(x4)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
