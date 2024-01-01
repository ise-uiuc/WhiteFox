
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.rand_like(x)
        x = F.dropout(x, p=0.2, training=False)
        x += F.dropout(x, p=0.5, training=False)
        return x
# Inputs to the model
x1 = torch.randn(1)
