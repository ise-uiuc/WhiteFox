
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.rand_like(x1, dtype=torch.float16)
        b1 = F.dropout(x1, p=0.3)
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 3)
