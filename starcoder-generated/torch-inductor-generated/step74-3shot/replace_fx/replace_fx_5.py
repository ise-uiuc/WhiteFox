
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1, p=0.1, training=torch.bool)
        return x2
# Inputs to the model
x1 = torch.randn(10)
