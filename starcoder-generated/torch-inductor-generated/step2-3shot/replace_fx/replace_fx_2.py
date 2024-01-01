
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        dropout_layer = torch.nn.Dropout(p=0.25)
        x2 = dropout_layer(x1)
        x3 = torch.dropout(x1, p=2)
        return x2 + x3
# Inputs to the model
x1 = torch.randn(10)
