
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        p1 = torch.nn.functional.dropout(input=x1, p=0.4, train=False, inplace=False)
        x3 = torch.nn.functional.dropout(input=x1, p=0.6, train=False, inplace=False)
        x4 = x1.reshape(1, 4, 4)
        x4 = torch.nn.functional.dropout(input=x4, p=0.4, train=False, inplace=False)
        x5 = x1.reshape(1, 2, 2, 2)
        return x3
# Inputs to the model
x1 = torch.randn(10)
