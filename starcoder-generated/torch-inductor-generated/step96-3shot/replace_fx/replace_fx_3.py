
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.dropout(x, training=True, inplace=True)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
