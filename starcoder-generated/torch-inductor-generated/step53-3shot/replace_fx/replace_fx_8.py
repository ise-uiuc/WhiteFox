
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        c = torch.nn.ReLU()
        x2 = c(F.dropout(x1, p=0.5)) # c(F.dropout(x1, p=0.5)) and c(F.dropout(x1, p=0.5)) will have the same replacement index, since it represents the same replacement.
        x3 = torch.rand(3, 3)
        return x2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        c = torch.nn.ReLU()
        x2 = c(F.dropout(x1, p=0.5))
        x3 = torch.rand(3, 3)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
