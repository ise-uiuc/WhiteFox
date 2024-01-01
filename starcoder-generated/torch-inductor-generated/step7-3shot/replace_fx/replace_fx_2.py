
class RandomModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a1 = torch.nn.functional.dropout(x1, p=0)
        a2 = torch.nn.functional.conv2d(a1, x1, a1, a1, mode='trilinear')
        a3 = a2.permute(1, 0, 2)
        a4 = a3.masked_fill(a1, 0)
        a5 = torch.rand_like(a1)
        return a1 * a1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
