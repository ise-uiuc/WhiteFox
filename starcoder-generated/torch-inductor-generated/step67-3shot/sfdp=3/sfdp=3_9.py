
class Attention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        mask = torch.triu(torch.ones_like(qk), diagonal=1).transpose(-2, -1).bool()
        qk.masked_fill(mask == 0, -1e4)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output, softmax_qk
attention = Attention(embed_dim)

# Inputs to the model
query = torch.randn(1, 8, 512)
key = torch.randn(1, 8, 512)
value = torch.randn(1, 8, 512)
scale_factor = 1.0 / math.sqrt(512)
dropout_p = 0.1
__output__, __softmax_qk__ = attention(query, key, value, scale_factor, dropout_p)
