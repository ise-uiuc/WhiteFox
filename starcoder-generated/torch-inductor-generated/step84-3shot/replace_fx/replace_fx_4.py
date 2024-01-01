
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = F.dropout(x, p=0.5)
        return y + y
# Inputs to the model
x = torch.randn(1, 1, 2)
