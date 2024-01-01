
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        r1 = F.dropout(x, p=0.5)
        r2 = r1.view(r1.size(0), -1)
        return r2
# Inputs to the model
x1 = torch.randn(10, 10)
