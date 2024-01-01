
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
    def forward(self, x):
        x = torch.nn.functional.dropout(x, 0.3)
        x = torch.nn.functional.dropout(x, p=0.3, training=True)
        x = self.dropout(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 2)
