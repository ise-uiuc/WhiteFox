
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        t0 = torch.matmul(x2, x1)
        t1 = torch.matmul(x2, x1)
        t2 = torch.matmul(x1, x2)
        t3 = t0 * t1
        t4 = t2 * t1
        return t3 + t4
# Inputs to the model
x1 = torch.randn(224, 224)
x2 = torch.randn(224, 224)
