
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2):
        x3 = F.dropout(x2, p=0.5)
        x4 = torch.rand_like(x2)
        return x4
# Inputs to the model
x2 = torch.randn(1, 2, 2)
