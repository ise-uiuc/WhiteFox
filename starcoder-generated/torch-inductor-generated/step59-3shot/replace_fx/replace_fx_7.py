
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = F.dropout(x1, p=0.5, inplace=False)
        x3 = F.dropout(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
