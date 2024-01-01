
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.2
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        x4 = x3 + x2
        x4 = torch.nn.functional.dropout(torch.nn.functional.dropout(x4, p=self.dropout_p), p=0.5)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
