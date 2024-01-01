
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4):
        f1 = F.linear(x1, x2)
        f2 = F.linear(x2, x3)
        f3 = F.linear(x3, x4)
        return f1 + f2 + f3
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
