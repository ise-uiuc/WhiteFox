
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(inplace=True)
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.5, training=False)
        x3 = torch.rand_like(x1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
