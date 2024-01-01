
class Model(torch.nn.Module):
    def __init__(self, p1, p2):
        super().__init__()
        t = torch.nn.Dropout(p1)
        self.dropout1 = torch.nn.Dropout(p2) # t is not captured in self.dropout1
        self.dropout2 = t
    def forward(self, x1):
        x2 = self.dropout1(x1)
        x3 = torch.rand_like(x2)
        x4 = self.dropout2(x3)
        return x2
p1 = 1
p2 = 0.6
# Inputs to the model
x1 = torch.randn(3, 3, 3)
