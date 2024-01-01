
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = F.dropout(x1)
        v2 = torch.rand_like(x1)
        return v1
# Inputs to the model
x2 = torch.randn(1, 2, 2)
