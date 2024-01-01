
class Model(torch.nn.Module):
    def __init__(self, p1):
        super().__init__()
        self.p1 = p1
    def forward(self, x1):
        x2 = x1 ** self.p1
        x3 = torch.nn.functional.dropout(x2)
        x4 = torch.rand_like(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
