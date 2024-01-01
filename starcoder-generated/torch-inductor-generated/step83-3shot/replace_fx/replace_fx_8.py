
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.reshape(x, (1,))
        x2 = torch.nn.functional.dropout(x, p=0.5)
        x3 = torch.reshape(x2, (1,))
        return x1 + x3
# Inputs to the model
x = torch.randn(2, 2)
