
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.nn.functional.dropout(x1, p=0.4)
        y = torch.rand_like(x1)
        return y
# Inputs to the model
x1 = torch.randn(1, 2, 2)
