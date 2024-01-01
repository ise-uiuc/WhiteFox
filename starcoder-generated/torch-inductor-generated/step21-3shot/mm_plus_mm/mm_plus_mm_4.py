
class Model(torch.nn.Module):
    def forward(self, x, y):
        y = y - 2
        y = y.clamp(min=0)
        return x * y[:,None]
# Inputs to the model
x = torch.randn(2, 2)
y = torch.Tensor([[-1, 1], [-2, 3]])
