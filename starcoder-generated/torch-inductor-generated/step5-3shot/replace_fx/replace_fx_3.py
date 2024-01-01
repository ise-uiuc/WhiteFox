
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.add(x1, x1)
        a2 = torch.dropout(input=a1, p=0.0, training=True)
        a3 = torch.mul(a1, a1)
        return torch.add(a2, a3)
# Inputs to the model
x1 = torch.randn(16, 8)
