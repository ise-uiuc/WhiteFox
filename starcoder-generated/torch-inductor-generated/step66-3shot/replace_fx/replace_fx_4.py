
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = x1.transpose(-1, -2)
        x2 = F.dropout(x2, p=0.5)
        x3 = x3 * x2
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 3)
