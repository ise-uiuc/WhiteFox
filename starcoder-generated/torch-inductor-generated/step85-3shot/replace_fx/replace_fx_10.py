
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = F.dropout(x, p=0.5)
        x2 = torch.rand_like(x, dtype=torch.int32)
        x3 = torch.nn.functional.dropout(x2, p=0.7)
        x4 = torch.rand_like(x2)
        x5 = F.dropout(x4, p=0.4)
        x6 = torch.rand_like(x2)
        x7 = torch.nn.functional.dropout(x2)
        x8 = torch.rand_like(x, dtype=torch.int32)
        x9 = torch.nn.functional.dropout(x8)
        x10 = torch.rand(1)
        x11 = F.dropout(x10, p=0.5)
        return x11
# Inputs to the model
x = torch.randn(8, 3)
