
# We use nn.Dropout2d in this example as it behaves similar to nn.functional.dropout
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout2d()
    def forward(self, x1, x2):
        # torch.nn.functional.dropout(x1, p=0.5)
        x3 = self.dropout(x1, p=0.5)
        # torch.nn.functional.dropout(x2, p=0.3)
        x4 = self.dropout(x2, p=0.3)
        # (x3 * x4).sum(-1)
        x5 = (x3 * x4).sum(-1)
        return x5
# Inputs to the model
x1 = torch.randn(1, 3, 3, requires_grad=True)
x2 = torch.randn(1, 3, 3, requires_grad=True)
