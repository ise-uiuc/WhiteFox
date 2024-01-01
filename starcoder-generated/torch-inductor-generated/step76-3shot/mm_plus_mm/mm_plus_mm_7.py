
class Model(nn.Module):
    def forward(self, x1, x2, x3):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x2, x3)
        h3 = torch.tanh(h1 + h2)
        h4 = torch.mm(x1, x3)
        h5 = torch.tanh(h4)
        return h3 + h5
# Inputs to the model
x1 = torch.randn(7, 7)
x2 = torch.randn(7, 7)
x3 = torch.randn(7, 7)
