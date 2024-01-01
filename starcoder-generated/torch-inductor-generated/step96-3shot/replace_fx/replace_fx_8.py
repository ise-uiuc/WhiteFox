
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 +1.0
        x2 = torch.nn.functional.dropout(x1, p=0.5)
        x3 = torch.cat([p for p in torch.split(x2, 1)], 0)
        return x1


# Inputs to the model
x1 = torch.randn(3, 2)
