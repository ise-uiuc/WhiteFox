
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        r1 = torch.nn.functional.dropout(x, p=0.3, training=True)
        r2 = torch.nn.functional.softmax(r1, dim=0)
        r3 = torch.mean(r2, dim=1)
        r4 = torch.sum(r3)
        return r4
# Inputs to the model
x1 = torch.randn(1, 4, 7, 8)
