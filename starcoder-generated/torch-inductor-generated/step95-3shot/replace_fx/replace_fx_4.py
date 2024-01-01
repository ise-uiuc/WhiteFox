
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        init = torch.rand_like(x1)
    def forward(self, x1):
        x2 = init
        x2 = F.dropout(x2, p=0.5) * x1
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
