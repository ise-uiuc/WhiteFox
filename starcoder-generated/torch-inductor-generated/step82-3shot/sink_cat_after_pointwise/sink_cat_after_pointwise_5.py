
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = torch.cat([x, x], dim=1)
        t1 = z.view(z.shape[0], -1, 2)
        t2 = z.view(z.shape[0], 2, -1)
        t3 = z.view(z.shape[0], -1)
        x = torch.cat([t1, t2, t3], dim=1)
        y = x.view(x.shape[0], -1, 4)  # Extra reshape
        w = y.unsqueeze(dim=2)        # Extra unsqueeze
        y = y.contiguous()            # Extra contiguous
        x = x.view(-1, 4)             # Extra reshape
        x = x + y                     # Extra plus
        x = x.sub(w)                  # Extra minus
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
