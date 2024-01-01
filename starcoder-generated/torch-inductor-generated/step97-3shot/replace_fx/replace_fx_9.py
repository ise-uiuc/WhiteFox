
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.2, training=True)
        return x2
        x3 = torch.rand_like(x2)
        return x1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
