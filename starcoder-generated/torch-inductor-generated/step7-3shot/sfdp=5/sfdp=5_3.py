
class Attention(torch.nn.Module):
    def __init__(self, attn_dropout_p: float):
        super().__init__()
        self.attn_dropout_p = attn_dropout_p
 
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, self.attn_dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the module
attention = Attention(0.0)

# Inputs to the module
query = torch.randn(5, 4, 17, 21)
key = torch.randn(5, 4, 15, 19)
value = torch.randn(5, 4, 15, 21)
