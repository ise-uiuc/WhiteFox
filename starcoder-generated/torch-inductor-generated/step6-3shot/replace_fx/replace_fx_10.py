
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v):
        a = v + v
        b = torch.nn.functional.dropout(a, p=0.0, training=True)
        return b
# Inputs to the model
x1 = torch.randn(2, 2)
