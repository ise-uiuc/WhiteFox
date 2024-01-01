
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.4899)
        x3 = F.dropout(x1, p=0.3983)
        x4 = torch.rand_like(x2)
        x5 = F.dropout(x2, p=0.212)
        return x5
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.4899)
        x3 = F.dropout(x1, p=0.3983)
        x4 = torch.rand_like(x2)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
