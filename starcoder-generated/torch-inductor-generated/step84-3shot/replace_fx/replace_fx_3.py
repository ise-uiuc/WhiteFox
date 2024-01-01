
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        d1 = torch.rand(())
        d2 = torch.rand(())
        self.d = (d1, d2)
    def forward(self, x):
        y = torch.sigmoid(F.dropout(x, p=0.49))
        v1, v2 = self.d
        return y + v1 + v2
# Inputs to the model
x = torch.randn(1, 1, 2)
