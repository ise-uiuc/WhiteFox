
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, X1, X2, mask=None):
        X = torch.cat([X1, X2], dim=1)
        q = torch.rand(1, 1, 24)
        k = torch.rand(1, 6, 24)
        v = torch.rand(1, 6, 32)
        qk = q@k.T/math.sqrt(k.shape[1])
        qk += mask
        attn_weight = torch.nn.Softmax(dim=-1)(qk)
        output = attn_weight @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
X1 = torch.randn(1, 2, 24)
X2 = torch.randn(1, 4, 24)
mask = torch.ones(1, 6).triu(1)
