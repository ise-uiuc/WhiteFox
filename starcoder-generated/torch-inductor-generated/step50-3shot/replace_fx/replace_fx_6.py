
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.randn(32, 32)
        x2 = F.dropout(x, p=0.5)
        return x2
# Inputs to the model
x1 = torch.randn(32, 32)
