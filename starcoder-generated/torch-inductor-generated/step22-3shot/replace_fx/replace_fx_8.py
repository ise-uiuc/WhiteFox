
class Model(torch.nn.Module):
    def __init__(self, p1, p2, p3):
        super(Model, self).__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
    def forward(self, x1):
        tmp = torch.nn.functional.dropout(x1, self.p1, True)
        x3 = x1 + tmp
        x2 = torch.randn_like(x1, self.p3)
        x4 = x2 * tmp
        return x1 + x2
p1 = 0.2
p2 = True
p3 = 0
# Inputs to the model
x1 = torch.randn(2, 2)
