
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        c1 = torch.matmul(x1, x2)
        c2 = torch.matmul(x1, x3)
        c3 = torch.matmul(x2, x3)
        o = torch.mean((c1+c2+c3)/(c1*c2*c3))
        return o
# Inputs to the model
x1 = torch.randn(256)
x2 = torch.randn(256)
x3 = torch.randn(256)
