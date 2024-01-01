
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.5, training=False)
        x2 = torch.rand_like(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 2, 2)
