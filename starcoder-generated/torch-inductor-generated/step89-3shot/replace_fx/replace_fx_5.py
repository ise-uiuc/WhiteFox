
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.nn.functional.dropout(x, p=0.7, inplace=True)
        x2 = torch.nn.functional.dropout(x, p=0.7, inplace=False)
        return torch.nn.functional.dropout(x2, p=0.7, inplace=True), torch.nn.functional.dropout(x1, p=0.7, inplace=True)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
