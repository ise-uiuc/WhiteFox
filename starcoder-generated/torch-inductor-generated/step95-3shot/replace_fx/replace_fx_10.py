
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = F.dropout(x1, p=0.1)
        x2 = F.dropout(x1, p=0.2)
        x3 = F.dropout(x1, p=0.25)
        x4 = F.dropout(x1, p=0.02)
        return (x1, x2, x3, x4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
