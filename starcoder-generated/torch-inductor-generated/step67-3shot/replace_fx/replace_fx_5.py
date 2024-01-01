
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.rand_like(x)
        w1 = x.size()[0]
        x = F.dropout(x, p=0.5)
        x = torch.rand_like(x)
        w2 = x.size()[0]
        x = F.dropout(x, p=0.5)
        return x
# Inputs to the model
x = torch.zeros(1)
