
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
    def forward(self, q, k, v1, mask):
        q = self.linear1(q)
        q = self.linear2(q)
        k = self.linear1(k)
        k = self.linear2(k)
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Input to the model
Q = torch.randn(1, embedding_dim, vocab_tokens_dim, vocab_tokens_dim)
K = torch.randn(1, embedding_dim, vocab_tokens_dim, vocab_tokens_dim)
V = torch.randn(1, embedding_dim, vocab_tokens_dim, vocab_tokens_dim)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
