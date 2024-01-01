
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x3):
        x4 = torch.nn.functional.dropout(x3, p=0.0)
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
