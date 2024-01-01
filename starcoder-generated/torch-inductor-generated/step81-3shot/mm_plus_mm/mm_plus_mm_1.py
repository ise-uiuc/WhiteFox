
class Model(torch.nn.Module):
    def forward(self, x, y):
        t1 = x + y
        t2 = x * y - y
        return t1 + t2
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(5, 5)
