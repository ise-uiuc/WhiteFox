
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        h1 = x1.matmul(x2)
        h2 = x2.matmul(x3)
        h3 = x3.matmul(x4)
        h4 = x4.matmul(x4)
        return h1 + h2 + h4
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
