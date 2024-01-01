
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = x1 + x2
        t2 = t1 + x2
        t3 = t2 + x2
        t4 = F.dropout(t3, p=0.5)
        x = t4 + x2
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = 1
