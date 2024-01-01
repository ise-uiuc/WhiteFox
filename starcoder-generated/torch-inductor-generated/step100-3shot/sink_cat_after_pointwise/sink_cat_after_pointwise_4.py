
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_layer = torch.nn.Dropout(p=0.5)
        self.weight = torch.randn(3, 4)
    def forward(self, x):
        x = x @ self.weight
        y = self.dropout_layer(x)
        return y.matmul(y.t())
# Inputs to the model
x = torch.randn(3, 4)
