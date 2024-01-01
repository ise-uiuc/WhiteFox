
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y2 = x * 0.4
        x = F.dropout(y2, p=0.5)
        y1 = x * 0.5
        return y1
# Inputs to the model
x = torch.randn(1, 2, 2)
