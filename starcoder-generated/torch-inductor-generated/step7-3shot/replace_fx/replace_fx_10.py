
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(inplace=False)
    def forward(self, x1):
        x = torch.nn.functional.dropout(x1, p=0.1, inplace=True, training=self.training)
        y = torch.nn.functional.dropout(x1, p=0.2, inplace=False)
        z = torch.nn.functional.dropout(x)
        return x + self.dropout(y + z)
# Inputs to the model
x1 = torch.randn(1, 2)
