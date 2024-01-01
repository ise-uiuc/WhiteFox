
class Model(torch.nn.Module):
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        v2 = torch.mm(x, y)
        return v1 + v2
# Inputs to the model
x = torch.randn(12, 12)
y = torch.randn(12, 12)
