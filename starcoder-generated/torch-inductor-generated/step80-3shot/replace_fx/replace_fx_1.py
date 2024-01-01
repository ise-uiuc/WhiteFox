
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        d = torch.nn.Dropout()
        print(d)
        x = F.dropout(x, p=0.5)
        return x
# Inputs to the model
x = torch.randn(1, 1, 2)
