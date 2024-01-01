
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=self.dropout_p)
        x3 = torch.rand_like(x2)
        x3 = torch.rand_like(x3)
        x4 = torch.nn.functional.dropout(x2 + x3, p=self.dropout_p)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
