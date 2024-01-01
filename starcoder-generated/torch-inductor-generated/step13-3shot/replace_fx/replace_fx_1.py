
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x)
        x2 = torch.rand_like(x)
        return x2 + x1
# Inputs to the model
x = torch.randn(150, 100)
