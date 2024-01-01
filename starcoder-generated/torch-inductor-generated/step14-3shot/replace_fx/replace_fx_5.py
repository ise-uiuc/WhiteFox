
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.tanh = torch.nn.Hardtanh()
    def forward(self, x1):
        x2 = self.tanh(x1)
        x3 = torch.rand_like(x1)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
