
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.rand_like(x1)
        x4 = F.dropout(x1, p=0.5)
        x5 = F.dropout(x2, p=0.5)
        return (x4, x5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
