
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        y = torch.rand_like(x)
        output = self.dropout(y)
        return output
# Inputs to the model
torch.randn(2,3)
