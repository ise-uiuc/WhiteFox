
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
    def forward(self, x):
        if self.dropout.p!= 0.4:
            x = self.dropout(x)
        else:
            x = x + 1
        x = torch.rand_like(x)
        x = torch.nn.functional.dropout(x, p=0.4)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
