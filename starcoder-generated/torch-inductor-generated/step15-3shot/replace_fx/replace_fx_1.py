
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y1 = torch.nn.functional.dropout(x1, p=0.001)
        y2 = torch.nn.functional.dropout(x1, p=0)
        return y1
# Inputs to the model
x1 = torch.randn(1, 2)
