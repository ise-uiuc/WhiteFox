
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        b1 = F.dropout(x1, p = 0.0, training = False)
        return b1
# Inputs to the model
x1 = torch.randn(1, )
