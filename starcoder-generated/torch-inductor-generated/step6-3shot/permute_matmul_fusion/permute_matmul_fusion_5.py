
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = x2.permute(2, 0, 1)
        v2 = torch.matmul(v1, x1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(2, 2, 2)
