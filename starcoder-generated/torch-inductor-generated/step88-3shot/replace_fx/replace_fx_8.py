
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.nn.functional.dropout(x1, p=0.5, training=True)
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        return x2 * x3
# Inputs to the model
x1 = torch.rand(1, 1)
