
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.abs(x1 * x2 * x3)
        return v1 + v1
# Inputs to the model
x1 = torch.randn(1, 5, 88, 88)
x2 = torch.randn(1, 5, 88, 88)
x3 = torch.randn(1, 5, 88, 88)
