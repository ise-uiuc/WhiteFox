
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        x = F.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(x, p=0.2)
        x = torch.nn.functional.dropout(self.dropout(x), p=0.2)
        x = F.dropout(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
