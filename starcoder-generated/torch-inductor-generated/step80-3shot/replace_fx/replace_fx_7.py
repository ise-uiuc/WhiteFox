
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        d1 = torch.rand(())
        d2 = torch.rand(())
        d3, d1 = torch.sort(d1, dim=0)
        return F.dropout(x, p=0.499) + d2 + d3
# Inputs to the model
x = torch.randn(1, 1, 2)
