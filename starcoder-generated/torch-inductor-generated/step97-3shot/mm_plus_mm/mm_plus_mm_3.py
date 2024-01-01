
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        h1 = x1.contiguous().view(x1.size()[0], -1)
        h2 = x2.contiguous().view(x2.size()[0], -1)
        h3 = x3.contiguous().view(x3.size()[0], -1)
        h4 = x4.contiguous().view(x4.size()[0], -1)
        return h1 + h2 + h3 * h4
# Inputs to the model
x1 = torch.randn(256, 4, 4)
x2 = torch.randn(256, 4, 4)
x3 = torch.randn(256, 4, 4)
x4 = torch.randn(256, 4, 4)
