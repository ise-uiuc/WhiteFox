
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x + torch.nn.functional.dropout(x, p=0.05)
# Inputs to the model
x = torch.randn(5)
