
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        q = x1
        k = x2
        att1 = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        att1 += -1e9 * (1 - (x2 == 0).float())
        att2 = F.softmax(att1, dim=-1)
        att2 = torch.dropout(att2, dropout_p, True)
        out = att2 @ v
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(bsz, l, 3)
x2 = torch.randn(bsz, r, 3)
x3 = torch.randn(bsz, r, 3)
