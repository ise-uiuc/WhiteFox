
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        p1 = torch.nn.functional.dropout(x, p=0.3)
        p2 = self.gelu(p1)
        return p2
# Inputs to the model
x1 = torch.randn(1, 28, 20)
