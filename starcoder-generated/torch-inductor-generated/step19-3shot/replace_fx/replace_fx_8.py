
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x)
        x2 = torch.rand_like(x)
        x3 = torch.nn.functional.dropout(x)
        return x1, x2, x3
# Inputs to the model
x = torch.randn(1, 1)
