
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, use_sig):
        x2 = torch.nn.functional.dropout(x, p=0.5)
        y1 = torch.matmul(use_sig(x), torch.transpose(use_sig(x2), 0, 1))
        y2 = torch.rand_like(x)
        z = y2 + y1
        z1 = F.softmax(z, dim=0)
        return z1
x = torch.randn(8,3)
