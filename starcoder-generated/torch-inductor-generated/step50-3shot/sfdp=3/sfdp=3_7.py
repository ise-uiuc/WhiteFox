
class Model(torch.nn.Module):
    def __init__(self, dim_model, num_heads, dropout_p):
        super().__init__()
        self.dim_model  = dim_model
        self.num_heads  = num_heads
        self.head_dim = dim_model / num_heads
        self.dropout_p = dropout_p

        # The query, key, and value layers are defined as separate classes in the original source.
        self.query_layer = torch.nn.Linear(self.dim_model, self.dim_model)
        self.key_layer   = torch.nn.Linear(self.dim_model, self.dim_model)
        self.value_layer = torch.nn.Linear(self.dim_model, self.dim_model)

        # The attention dropout is defined as class variable in the original source,
        # because it also is applied to the query, key, and value tensors at the same time.
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask):
        # Apply the linear layers to the query, key, and value features.
        q = self.query_layer(query)
        k = self.key_layer(key)
        v = self.value_layer(value)

        # Split the output of the linear layers into the specified number of heads,
        # and then pass them through different projections in the attention computation.
        q = self._reshape_and_transpose(q)
        k = self._reshape_and_transpose(k)
        v = self._reshape_and_transpose(v)

        # Scale the dot product of the query and key tensors by the square root of the head dimension.
        scale_factor = self.head_dim ** -0.5
        scores      = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

        # Mask the score tensor with the specified mask value,
        # and then apply softmax to the scores tensor along the last dimension.
        scores = scores.masked_fill(mask == 0, -1e9)
        probs  = scores.softmax(dim=-1)

        # Apply dropout to the dropout rates for the query, key, and value tensors.
        q = self.dropout(q)
        k = self.dropout(k)
        v = self.dropout(v)

        # Compute the dot product of the dropout output of the value tensor
        # and the dropout output of the softmax output of the attention scores.
        attn_out = torch.matmul(probs, v)

        # Reshape and transpose the output of the attention dot product
        # from the shape of [B, H, S, D] to the shape of [B, S, D].
        return attn_out.reshape(q.shape)

    def _reshape_and_transpose(self, x):
        