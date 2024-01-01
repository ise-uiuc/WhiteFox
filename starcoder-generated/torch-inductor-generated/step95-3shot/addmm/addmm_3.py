
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.randn(10, 10, requires_grad=True)
        with torch.no_grad():
            self.x2 = torch.randn(10, 10)
    def forward(self, x2):
        q = torch.mm(x2, self.x1)
        return q + self.x2
# Inputs to the model
x2 = torch.randn(5, 5, requires_grad=True)
