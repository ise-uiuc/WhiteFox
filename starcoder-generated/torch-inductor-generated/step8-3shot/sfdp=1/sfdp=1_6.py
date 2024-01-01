
class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.in_proj_weight = nn.Parameter(torch.empty(3*embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3*embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.r_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[0:embed_dim, :])
        xavier_uniform_(self.in_proj_weight[embed_dim:2*embed_dim, :])
        xavier_uniform_(self.in_proj_weight[2*embed_dim:, :])
        nn.init.constant_(self.in_proj_bias[0:embed_dim], 0.)
        nn.init.constant_(self.in_proj_bias[embed_dim:2*embed_dim], 0.)
        nn.init.constant_(self.in_proj_bias[2*embed_dim:], 0.)
        xavier_uniform_(self.r_proj.weight)
        nn.init.constant_(self.r_proj.bias, 0.)
        self.out_proj.reset_parameters()

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        