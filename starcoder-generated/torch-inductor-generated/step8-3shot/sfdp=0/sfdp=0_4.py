
class SingleHeadAttention(torch.nn.Module):
    def __init__(self, dim: int, d_model: int, d_keys: int):
        super().__init__()
        # Shape: `[d_keys per head, d_model]`.
        self.w_query = torch.nn.Linear(d_model, dim)
        # Shape: `[d_keys per head, d_model]`.
        self.w_keys = torch.nn.Linear(d_model, d_keys)
        # Shape: `[d_keys per head, d_model]`.
        self.w_values = torch.nn.Linear(d_model, d_keys)
        self.dim = dim

    def forward(self, x1, x2, mask=None):
        d_keys = self.w_keys.out_features // self.dim
        query = apply_chunking_to_forward(self.w_query.forward, self.chunk_size, self.seq_len_dim, query=x1)
        key = apply_chunking_to_forward(self.w_keys.forward, self.chunk_size, seq_len_dim=self.seq_len_dim, key=x2)
        value = apply_chunking_to_forward(self.w_values.forward, self.chunk_size, self.seq_len_dim, value=x2)
        # Shape: `[heads, num queries, num key-value pairs, d_keys per head]`.
        scaled_dot_product = torch.einsum("... qk,... k ->... qk", query, key)
        if mask is not None:
            # Shape: `[num queries, num key-value pairs]`
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == scaled_dot_product.shape[-1], f"mask has incorrect dimensions. Pass shapes {mask.shape} and {scaled_dot_product.shape}."  # noqa
            # Shape: `[num queries, num key-value pairs, 1]`
            bias = torch.full_like(mask, -1e4)
            scaled_dot_product = scaled_dot_product.masked_fill(mask.unsqueeze(-2), bias).type_as(scaled_dot_product)
        inv_sqrt_d_keys = 1 / math.sqrt(d_keys)
        # Shape: `[heads, num queries, num key-value pairs, d_keys per head]`.
        scaled_dot_product = scaled_dot_product * inv_sqrt_d_keys
        # Shape: `[heads, num queries, num key-value pairs, d_keys per head]`.
        attention_weights = scaled_dot_product.softmax(dim=-1)
        # Shape: `[heads, num queries, d_keys per head]`.
        attention_output = torch.einsum("... qkp,... pk ->... qk", attention_weights, value)
        return attention_output

class EncoderLayer1(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, n_heads: int, d_keys: int, chunk_size: int, seq_len_dim: int):
        super().__init__()
        self.norm_one = torch.nn.LayerNorm(model_dim)
        self.norm_two = torch.nn.LayerNorm(model_dim)
        # Multi-head attention.
        self.attn = MultiHeadAttention(model_dim, n_heads, d_keys)
        self.ln_one = torch.nn.LayerNorm(model_dim)
        # Feed-forward network.
        self.ffn = PointwiseFeedForwardNet(model_dim, inner_dim)
        self.ln_two = torch.nn.LayerNorm(model_dim)
 
    def forward(self, x, mask=None):
        residual = x
        x = self.norm_one(x)
        x = self.ln_one(x)
        x = self.attn(query=x, key=x, value=x, mask=mask)
        x = x + residual
        residual = x
        x = self.norm_two(x)
        x = self.ln_two(x)
        x = self.ffn(x)
        x = x + residual
        return x

class Encoder(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, n_layers: int):
        super().__init__()
        # A single encoder layer.
        # 1. Create the multi-head attention layer.
        # 2. Create the layernorm layer and pass it as the `sublayer` argument to
        #    the multi-head attention layer.
        # 3. Pass the input tensor, the output of the previous layer and the attention
        #    mask to the multi-head attention layer.
        # 4. Apply the output of the attention layer on the input tensor to the
        #    layernorm layer to normalize the output of the self-attention layer.
        # 5. Apply the normalized tensor to another pointwise MLP.
        #    Take the output of this MLP and add it to the output of the previous layer.
        #    Apply the output the of MLP with layernorm and pass the output to the previous
        #    layer as well.
        self.layers = torch.nn.ModuleList([
            EncoderLayer1(model_dim, inner_dim, n_heads, d_keys, chunk_size, seq_len_dim)
            for i in range(n_layers)
        ])
        # LayerNorm after the last encoder layer.
        self.norm = torch.nn.LayerNorm(model_dim)
 
    def forward(self, x, mask=None):
        output = x
        for layer in self.layers:
            output = layer(output, mask)
        output = self.norm(output)
        return output

# Initializing the model
m = Encoder(model_dim, inner_dim, n_layers)

# Inputs to the model
x1 = torch.randn(1, 1, model_dim, 64, 64)
x2 = torch.randn(1, 1, model_dim, 64, 64)
mask = torch.randn(1, 1, 1, 64, 64)
