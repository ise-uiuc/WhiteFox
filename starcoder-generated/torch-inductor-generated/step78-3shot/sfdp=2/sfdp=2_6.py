
class Model(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.dropout_p = 0.5

    def transpose_for_scores(self, x):
        new_m = x.view(x.size(0), x.size(1), self.num_attention_heads, self.attention_head_size)
        return new_m.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), self.attention_head_size)

    def forward(self, q, k, v, mask_k=None):
        query = self.transpose_for_scores(self.query(q))
        key = self.transpose_for_scores(self.key(k))
        value = self.transpose_for_scores(self.value(v))

        scaled_qk = torch.matmul(query, key.transpose(-2, -1)) * (1 / np.sqrt(self.attention_head_size))

        if mask_k:
            mask_k = mask_k.repeat(key.shape[1], key.shape[0], 1)
            scaled_qk = scaled_qk.masked_fill(mask_k, value=-np.inf)

        softmax_qk = torch.softmax(scaled_qk, dim=-1)

        dropout_qk = torch.nn.functional.dropout(softmax_qk,
                                                 p=self.dropout_p,
                                                 training=self.training)

        return torch.matmul(dropout_qk, value).view(-1, query.size(0), self.d_model)

bert_dim = 32
d_model = 64
num_attention_heads = 8
attention_head_size = 16

m = Model(d_model)

dropout_p = 0.5

q = torch.randn(2, 3, bert_dim)
k = torch.randn(2, 4, bert_dim)
v = torch.randn(2, 4, bert_dim)
mask_k = torch.randn((4, 1)) > 0
