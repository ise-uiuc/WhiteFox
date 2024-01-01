
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = x2.permute(1, 0, 2)
        v2 = x1.permute(0, 2, 1)
        v3 = torch.matmul(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 3, 2)
