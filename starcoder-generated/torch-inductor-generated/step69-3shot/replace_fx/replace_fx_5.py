
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = F.dropout(x2, p=0.5, inplace=False)
        x4 = F.dropout(x1 + x2, p=0.5, inplace=False)
        x5 = F.dropout(x4, p=0.5, training=False)
        return x5
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.nn.functional.dropout(x2, p=0.5, inplace=False)
        x4 = torch.nn.functional.dropout(x1 + x2, p=0.5, inplace=False)
        x5 = torch.nn.functional.dropout(x4, p=0.5, training=False)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
