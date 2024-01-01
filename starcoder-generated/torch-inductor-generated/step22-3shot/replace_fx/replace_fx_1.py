
class model(torch.nn.Module):
    def __init__(self, a1, b1):
        super().__init__()
        self.a1 = torch.tensor([a1], dtype=torch.float)
        self.b1 = torch.tensor([b1], dtype=torch.float)
    def forward(self, x1):
        x2 = torch.einsum("ij,j->i", x1, F.dropout(self.a1))
        x3 = F.dropout(F.sigmoid(x2))
        x4 = torch.einsum("ij,j->i", x1, F.dropout(self.b1))
        x5 = F.dropout(F.sigmoid(x4))
        x6 = x3 + x5
        x7 = torch.cat((x3, x4), dim=0)
        return x5
a1 = 1
b1 = 2
# Inputs to the model
x1 = torch.randn(2, 2)
