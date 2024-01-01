
class Model(torch.nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, q, k, v, mask=None, bias=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= np.sqrt(self.hidden_size)

        if bias is not None:
            scores += bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if self.training:
            p_attn = dropout(p_attn, self.dropout_p)
        output = torch.matmul(p_attn, v)

        return output, p_attn

# Initializing the model
embed_dim, num_heads = 32, 4
dropout_p = 0
model = Model(embed_dim)
q = torch.randn(6, 10, embed_dim)
k = torch.randn(11, 10, embed_dim)
v = torch.randn(11, 10, embed_dim)
__output__, __p_attn__ = model(q, k, v, dropout_p)

