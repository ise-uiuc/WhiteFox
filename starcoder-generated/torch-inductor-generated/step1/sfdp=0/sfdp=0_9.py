
class Model(torch.nn.Module):
    def __init__(self, emb, n_heads, d_model):
        super().__init__()
        self.emb = emb
        self.num_heads = n_heads
        self.d_model = d_model
        self.proj_query = torch.nn.Parameter(torch.randn(self.num_heads, self.d_model, self.emb))
        self.proj_key = torch.nn.Parameter(torch.randn(self.num_heads, self.d_model, self.emb))

    def forward(self, q, k, v, mask=None, inv_scale=None):
        bsz = q.shape[0]

        w = torch.matmul(q, torch.transpose(k, -1, -2))
        w = w / inv_scale.view(-1, 1, 1)
        if mask is not None:
            w = torch.where(mask, w, torch.full_like(w, float('-inf')))
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.nn.functional.dropout(w, 0.1, True)  # Always dropout
        a = torch.matmul(w, v)
        return a

# Initializing the model
m = Model(emb=128, n_heads=8, d_model=128)

# Inputs to the model
q = torch.randn((1, 4, 128))
k = torch.randn((1, 8, 128))
v = torch.randn((1, 8, 128))
inv_scale = torch.randn((1, 128))
mask = torch.randn((1, 4, 8))
