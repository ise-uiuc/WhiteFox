
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z1 = torch.nn.functional.gelu(0.1)
        z2 = z1
        w1 = torch.nn.functional.dropout(z2)
        return z2
# Inputs to the model
x = torch.randn(3, 4, 5)
