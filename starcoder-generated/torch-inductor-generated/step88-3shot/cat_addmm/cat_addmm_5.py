
class Model(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.query_linear = nn.Linear(D_Model, D_Model)
        self.key_linear = nn.Linear(D_Model, D_Model)
        self.value_linear = nn.Linear(D_Model, D_Model)
        self.combine_heads = nn.Linear(D_Model, D_Model)

    def forward(self, query, key, value, mask=None):
        # l(W_Q.Q)
        query_linear_out = self.query_linear(query)# r(W_K.K)
        # batch x num_heads x seq_len_q x D_K
        key_head_out = self.key_linear(key)._split_batch(self.h)
        # batch x num_heads x Seq_len_q x D_K
        value_head_out = self.value_linear(value)._split_batch(self.h)
        # perform dot product
        score = torch.einsum('bqnd,bknd->bhqn', [query_linear_out, key_head_out])
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        # perform softmax
        attention = torch.softmax(score, dim=-1)

        # r(W_V.V)
        value_head_out = self.value_linear(value)._split_batch(self.h)
        # attention x batch x num_heads x D_V
        weighted = torch.einsum('bhqn,bknd->bqnd', [attention, value_head_out])
        # concatenate heads
        out = weighted._combine_batch(self.h)
        # l(W_O.attention)
        out = self.combine_heads(out)
        return out
# Inputs to the model
x = torch.randn(2, 3)
