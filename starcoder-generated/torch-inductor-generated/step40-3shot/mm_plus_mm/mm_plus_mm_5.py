
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2)
        v2 = torch.mm(x1, x1)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(23, 2)
x2 = torch.randn(2, 12)
