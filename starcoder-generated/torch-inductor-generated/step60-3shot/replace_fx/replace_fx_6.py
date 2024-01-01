
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0.5, training=True)
        a2 = F.dropout(x, p=0.5)
        return a2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.nn.functional.dropout(x, p=0.5, training=True)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
