
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x + x
        return F.dropout(z, p=0.5)
# Inputs to the model
x = 1
