
class Model(torch.nn.Module):
    def __init__(self, num_heads: int, dropout_p: float, use_mask: bool):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.use_mask = use_mask

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                                 q_mask: Optional[torch.Tensor] = None, 
                                 k_mask: Optional[torch.Tensor] = None):
        attn = torch.matmul(q, k.transpose(-2, -1))
        if self.use_mask:
            attn = attn.masked_fill(q_mask[:, :, None, None] == 0, -1e4)
            attn = attn.masked_fill(k_mask[:, None, :, None] == 0, -1e4)
        attn = attn.softmax(dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout_p, training=self.training)
        out = attn @ v
        return out

# Initializing the model
m = Model(num_heads=1, dropout_p=0.5)

# Inputs to the model
q = torch.randn(4, 2, 20)
k = torch.randn(4, 2, 30)
v = torch.randn(4, 2, 30)
