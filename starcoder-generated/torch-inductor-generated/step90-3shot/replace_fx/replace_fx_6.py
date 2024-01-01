
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand(1, 2, 2)
        x3 = torch.rand(1, 2, 2)
        x3, _ = F.dropout(x3, p=1.0, training=self.training)
        y1 = x3 * x2
        return y1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
