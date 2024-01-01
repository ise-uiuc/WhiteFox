
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value, attn_mask, dropout_p):
        q = query
        k = key.transpose(-2, -1)
        qk = q @ k / math.sqrt(q.size(-1))

        # The mask should have the same type as the query
        # Transformer only supports mask for attn_mask (int).
        # We can convert the mask to attn_mask by subtracting a large number.
        if attn_mask is not None and attn_mask.dtype == torch.int8:
            attn_mask = attn_mask.to(torch.get_default_dtype())
            attn_mask = (attn_mask - 1) * 1e9
            query_batch, query_seq_length = q.size()[:2]
            attn_keys_per_head = query_batch * query_seq_length
        else:
            attn_keys_per_head = None

        qk = qk.masked_fill(attn_mask, -1e9)

        attn_weight = self._attn_weight_fn(qk, attn_keys_per_head)
        attn_weight = torch.nn.functional.dropout(
            attn_weight, p=dropout_p, training=self.training)
        output = attn_weight @ value
        return output

    @staticmethod
    def _attn_weight_fn(query_key, attn_keys_per_head: Optional[int]):
        return F.softmax(query_key, dim=-1)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.tensor(np.random.rand(2, 1, 64))
key = torch.tensor(np.random.rand(2, 3, 64))
value = torch.tensor(np.random.rand(2, 3, 64))
attn_mask = torch.ones(2, 1)
dropout_p = 0.5
