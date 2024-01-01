
class Model(torch.nn.module.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = F.dropout(x, p=0.1, training=True)
        x = torch.rand_like(x)
        x = F.dropout(x, p=0.5)
        x = F.dropout(x, p=0.5)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
