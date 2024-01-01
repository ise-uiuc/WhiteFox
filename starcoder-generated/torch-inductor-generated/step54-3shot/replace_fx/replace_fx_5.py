
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        l1 = x1.view(-1, 160, 7, 7)
        r1 = x2.view(-1, 192, 4, 4)
        x3 = torch.sum(l1, dim=[2, 3])
        x4 = torch.sum(r1, dim=[2, 3])
        x5 = x3 + x4
        return x5
# Inputs to the model
x1 = torch.randn(2, 3, 10, 11)
x2 = torch.randn(2, 3, 21, 22)
