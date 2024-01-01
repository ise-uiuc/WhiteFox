
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        s1 = torch.matmul(v1, x2)
        v2 = x2.permute(0, 2, 1)
        s2 = torch.matmul(x1, v1)
        return s1 + s2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
model1 = Model1()
