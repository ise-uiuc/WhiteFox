
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.nn.functional.dropout(x, p=0.25)
        y = torch.rand_like(x)
        return y
# Inputs to the model
x1 = torch.rand((4, 6))
