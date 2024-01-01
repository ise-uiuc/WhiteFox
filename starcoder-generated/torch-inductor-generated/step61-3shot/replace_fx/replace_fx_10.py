
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a1 = torch.nn.functional.dropout(x, p=0)
        return 1
# Inputs to the model
x = torch.randn(2)
