
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.5, training=False)
        y1 = torch.nn.functional.dropout(x1, p=0.7, training=True)
        y2 = torch.rand_like(x1)
        return t1
# Inputs to the model
x1 = torch.randn(1, 2)
