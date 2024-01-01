
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        return v1 * 2
# Inputs to the model
x1 = torch.randn(11, 11)
x2 = torch.randn(11, 11)
