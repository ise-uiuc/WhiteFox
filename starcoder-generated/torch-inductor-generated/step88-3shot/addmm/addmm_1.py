
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2_ = torch.nn.functional.dropout(x1, p=0.5, training=True)
        return x1 * x2_
# Inputs to the model
x1 = torch.randn(3, 3)
