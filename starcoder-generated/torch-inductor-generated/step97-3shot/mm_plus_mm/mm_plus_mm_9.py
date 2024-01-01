
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        d = x1.matmul(x2) + x2.matmul(x3) + x3.matmul(x4)
        l = x4.matmul(x2)
        return d + l
# Inputs to the model
x1 = torch.randn(20, 20)
x2 = torch.randn(20, 20)
x3 = torch.randn(20, 20)
x4 = torch.randn(20, 20)
