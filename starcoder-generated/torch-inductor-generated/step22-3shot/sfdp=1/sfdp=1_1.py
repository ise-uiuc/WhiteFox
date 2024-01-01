
class Model(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_qkv, scale_attn_weights=False, dropout_p=0.0,
                 attn_mask_mode="add"):
        super().__init__()
        self.linear_q = torch.nn.Linear(d_model, n_heads * d_qkv)
        self.linear_k = torch.nn.Linear(d_model, n_heads * d_qkv)
        self.linear_v = torch.nn.Linear(d_model, n_heads * d_qkv)
        self.scale_attn_weights = scale_attn_weights
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.attn_mask_mode = attn_mask_mode
        self.sqrt_dk = np.sqrt(d_qkv)

    def forward(self, x1, x2):
        q, k, v = self.linear_q(x1), self.linear_k(x2), self.linear_v(x2)
        q, k, v = q.reshape(*q.shape[:-1], -1, q.shape[-1]), k.reshape(*k.shape[:-1], -1, k.shape[-1]), \
            v.reshape(*v.shape[:-1], -1, v.shape[-1])
        attn = torch.matmul(q, k.transpose(2, 3))
        if self.scale_attn_weights:
            attn = attn / self.sqrt_dk
        if self.attn_mask_mode == "add":
            attn = attn + attn_mask
        if self.attn_mask_mode == "mul":
            attn = attn * attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)
        attn = attn.flatten(2, 3)
        return attn

# Initializing the model
m = Model(d_model=256, n_heads=16, d_qkv=64, scale_attn_weights=False)

# Inputs to the model
x1 = torch.randn(1, 16, 256)
x2 = torch.randn(2, 16, 256)
attn_mask = None
