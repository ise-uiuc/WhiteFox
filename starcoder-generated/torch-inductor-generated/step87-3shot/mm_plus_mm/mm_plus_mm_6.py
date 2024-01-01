
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x2, x3)

        h3 = torch.nn.ReLU()(h1)
        h4 = torch.nn.ReLU()(h2)
        h5 = torch.nn.ReLU()(h3)

        h6 = torch.mm(h4, h5)

        h7 = h6 + h6 - h6

        h8 = torch.mm(h7, h7)
        return h8
# Inputs to the model
x1 = torch.randn(16, 4)
x2 = torch.randn(4, 16)
x3 = torch.randn(16, 32)
