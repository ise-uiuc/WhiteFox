
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.rand_like(x, dtype=torch.long)
        x2 = torch.nn.functional.dropout(x, p=0.2, training=False)
        return x2
# Input to the model
x1 = torch.randn(75, 80)
