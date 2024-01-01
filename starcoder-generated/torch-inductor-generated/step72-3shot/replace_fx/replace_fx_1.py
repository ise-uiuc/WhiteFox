
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = F.dropout(x1, p=0.5)
        x4 = F.dropout(x2, p=0.5)
        x5 = torch.rand_like(x1) # rand_like and dropout invoke the same underlying Fairseq function! x5 should end up with the same shape as x1
        x6 = torch.rand_like(x2)
        return x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
