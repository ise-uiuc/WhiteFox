
class Model(torch.nn.Module):
    def __init__(self, n_head, dim, d_ff):
        super().__init__()
        self.n_head = n_head
        self.dim = dim
        self.d_ff = d_ff

        self.in_linear = torch.nn.Linear(dim, d_ff)
        self.ff_linear = torch.nn.Linear(d_ff, dim)
        self.out_linear = torch.nn.Linear(dim, dim)

    def forward(self, q, k, v, attn_mask):
        n_query = q.size(1)
        q = self.in_linear(q)
        k = self.out_linear(self.in_linear(k))
        v = self.out_linear(self.in_linear(v))
        q *= self.dim**-0.5

        q, k, v = split_heads(q, k, v, self.n_head)

	# Padding for attention mask, because batch_size can be different
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        attn_mask = attn_mask.repeat(1, self.n_head, n_query, 1).unsqueeze(1)

        q, k, v = attn_core(q, k, v, attn_mask)

        q = merge_heads(q, self.n_head)
        q = self.ff_linear(q)
        q = self.out_linear(q)
        return q

# Initializing the model
n_head = 8
dim = 64
d_ff = 1024
n_query = 100
n_memory = 200
x1 = torch.randn(1, n_query, dim)
memory = torch.randn(1, n_memory, dim)
attn_mask = torch.tril(torch.ones((1, n_query, 1, n_memory)), 0).cuda()

m = Model(n_head, dim, d_ff)
