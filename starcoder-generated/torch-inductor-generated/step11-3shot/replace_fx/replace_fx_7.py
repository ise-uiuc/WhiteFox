
class Model(torch.nn.Module):
    def __init__(self, d=0.5):
        super().__init__()
        self.d = d
    def forward(self, x):
        c1 = torch.nn.functional.dropout(x, p=self.d)
        return 1
# Inputs to the model
x1 = torch.randn(1)
x = 1
