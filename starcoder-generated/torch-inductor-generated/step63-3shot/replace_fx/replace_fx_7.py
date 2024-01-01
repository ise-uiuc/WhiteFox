
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = F.linear(x1, x2)
        x4 = F.dropout(x3, p=0.5)
        x5 = torch.rand_like(x3)
        x6 = torch.nn.functional.dropout(x5, p=0.4)
        return x6
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
