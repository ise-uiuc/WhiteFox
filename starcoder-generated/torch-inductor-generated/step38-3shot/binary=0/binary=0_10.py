
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = x1 * x2
        return (v1 + other)
# Inputs to the model
x1 = torch.randn(1, 2, 3, 2)
x2 = torch.randn(1, 2, 3, 2)
