
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.mm(x1, x2)
    def forward(self, x1, x2):
        #v1 = torch.mm(x1, x2)
        return torch.cat([self.v1.T, self.v1.T, self.v1.T, self.v1.T, self.v1.T, self.v1.T, self.v1.T, self.v1.T], 1)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
