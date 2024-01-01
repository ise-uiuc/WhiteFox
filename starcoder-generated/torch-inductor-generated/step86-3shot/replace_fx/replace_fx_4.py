
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x = x1 + 1.0
        x = F.dropout(x, training=True)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 2)
