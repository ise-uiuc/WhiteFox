
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = F.dropout(x1, p=0.5)
        x2 = F.dropout(x1, p=0.5)
        t1 = torch.randn_like(x1)
        t2 = torch.randn_like(x1)
        return (x1, t2, t1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
