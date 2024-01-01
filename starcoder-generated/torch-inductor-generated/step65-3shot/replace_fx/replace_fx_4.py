
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5)
        x3 = torch.rand_like(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
