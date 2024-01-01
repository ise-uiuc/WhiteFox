
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = x[0]
        x = F.dropout(x)
        return x
# Inputs to the model
x1 = torch.randn(1)
