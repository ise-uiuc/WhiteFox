
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        r1 = torch.nn.functional.dropout(x, p=0.3, training=True)
        r2 = torch.cat([r1, y], dim=1)
        r3 = torch.nn.functional.softmax(r2, dim=0)
        r4 = torch.sum(r3)
        r5 = r4.reshape(-1, 3)
        return r5[0][1:]
# Inputs to the model
x = torch.randn(4, 7, 8)
y = torch.randn(4, 7, 8)
