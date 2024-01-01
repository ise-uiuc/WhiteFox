
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.5)
        x2 = x1.reshape((x1.shape[0], -1))
        return x2
# Inputs to the model
x1 = torch.randn(128,768)
