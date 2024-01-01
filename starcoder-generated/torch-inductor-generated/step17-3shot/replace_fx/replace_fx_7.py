:
#   Note: `dropout_p` is a parameter of forward() to make the pattern more meaningful.
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, p):
        c1 = torch.nn.functional.dropout(x1, p=p)
        return c1
# Inputs to the model
x1 = torch.randn(1)
p = torch.tensor(torch.nn.functional.dropout(torch.rand(1)))
