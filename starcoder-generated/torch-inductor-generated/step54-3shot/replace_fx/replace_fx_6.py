
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x1):
        r = torch.nn.functional.dropout(x, p=0.3, training=False)
        r1 = torch.nn.functional.softmax(r, dim=0)
        s = torch.mean(r1, dim=1)
        r2 = torch.sum(s)
        r3 = torch.cat([torch.relu(x), x1], dim=0)
        r4 = torch.tensor([1.0, 2.3, 3.2, 4.2, 5.2, 6.3])
        return torch.mul(r2, r3)
# Inputs to the model
# Shape of (1, 3, 4)
x = torch.randn(1, 3, 4)
# Shape of (1, 3, 1)
x1 = torch.randn(1, 3, 1)
