
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = (math.sqrt(5) + 1) / 2
        p = t / (t + 1)
        x = F.dropout(x, p=p)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
