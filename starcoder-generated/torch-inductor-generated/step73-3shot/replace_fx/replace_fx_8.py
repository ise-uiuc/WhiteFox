
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x2 = torch.randn(1, 2, 2)
        x3 = torch.nn.functional.dropout(x) # dropout 1
        y = x3 + x # dropout 2
        y1 = x2 + y1 # dropout 4
        y2 = torch.nn.functional.dropout(y1, p=0.8) # dropout 5
        y3 = torch.rand_like(y1) # rand
        y4 = F.dropout(y3, p=0.8) # dropout 6
        y5 = torch.rand_like(y1) # rand
        y6 = F.dropout(y5, p=0.1) # dropout 7
        return y6
# Inputs to the model
x = torch.randn(8, 3)
