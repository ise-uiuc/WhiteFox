
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        b1 = torch.nn.functional.dropout(x, p=0.2, training=True)
        b2 = torch.nn.functional.dropout(x, training=False)
        return 1
# Inputs to the model
x = torch.randn(1)
