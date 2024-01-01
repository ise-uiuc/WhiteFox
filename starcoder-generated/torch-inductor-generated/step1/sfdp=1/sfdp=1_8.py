
class Model(torch.nn.Module):
    def __init__(self, nheads, dim_head, dim, dropout_p=0.5):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)
        self.m = torch.nn.Linear(dim, dim, bias=False)
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.m(x)
        attn = (q @ k.transpose(-2, -1)).div(self.dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = attn @ v
        return x

# Initializing the model
m = Model(32, 16, 768, 0.5)

# Inputs to the model
x = torch.randn(2, 768, requires_grad=True)

