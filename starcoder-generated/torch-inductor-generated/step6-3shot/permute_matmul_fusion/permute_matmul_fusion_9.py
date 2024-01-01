
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2.permute(2, 0, 1)
        v3 = torch.matmul(v1, v2)
        return v1
# Inputs to the model
x1 = torch.randn(3, 2, 2)
x2 = torch.randn(4, 2, 2)
