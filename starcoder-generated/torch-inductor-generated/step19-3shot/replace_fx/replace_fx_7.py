
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        w1 = torch.rand_like(x, dtype=torch.float)
        y = torch.nn.functional.dropout(x, p=0.8, training=True)
        z = w1 + y
        return z
# Inputs to the model
x = 1
