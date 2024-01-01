
class MultiHeadAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.head_dim = emb // heads

        assert (
            self.head_dim * heads == emb
        ), "Embedding dimension must be divisible by number of heads"

        self.qkv = nn.Linear(emb, emb * 3, bias=False)
        self.fc = nn.Linear(emb, emb)

    def forward(self, q, k, v):
        head_dim = self.head_dim
        assert q.shape[1] == k.shape[1] == v.shape[1]

        # Calculate q, k, and v
        qkv = (
            self.qkv(q)
          .reshape(q.shape[0], q.shape[1], 3, self.heads, head_dim)
          .permute(2, 0, 3, 1, 4)
        )
        k, v = (
            k.reshape(k.shape[0], k.shape[1], self.heads, head_dim).permute(0, 2, 1, 3),
            v.reshape(v.shape[0], v.shape[1], self.heads, head_dim).permute(0, 2, 1, 3),
        )

        q, k, v= list(map(lambda t: rearrange(t, 'b h n d -> b n (h d)'),'q k v'.split()))

        # Dot-product attention
        dots = torch.matmul(q, k) * (head_dim ** -0.5)
        
        # Softmax
        attn = dots.softmax(dim=-1)

        # Dropout
        attn = nn.functional.dropout(attn, p=dropout_p, training=self.training)

        # # Out projection
        out = torch.matmul(attn, v)

        # Restore original shape
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.fc(out)

        return out

# Inputs to the model
q = torch.randn(1, 64, 2000)
k = torch.randn(1, 64, 1000)
v = torch.randn(1, 64, 1000)
out = MultiHeadAttention(2000, 10)(q, k, v)

