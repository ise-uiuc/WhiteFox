
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(0.0)
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.randn(10)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2)
