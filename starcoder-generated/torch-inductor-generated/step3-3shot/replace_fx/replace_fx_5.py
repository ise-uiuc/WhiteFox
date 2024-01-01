
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v2 = torch.nn.functional.dropout(x1)
        z1 = torch.nn.functional.dropout(x1, p=0.8)
        w1 = torch.rand_like(x1, dtype=torch.float)
        z2 = v2 + w1
        t2 = z1 + z2
        y1 = torch.nn.functional.gelu(t2)
        return y1
# Inputs to the model
x1 = torch.randn(3, 3, 3)
x2 = x1.reshape(3, 3, 1)
