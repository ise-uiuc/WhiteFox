
class Model(torch.nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

    def forward(self, enc_input):
        query = enc_input.transpose(-2, -3)

        query, key, value = self.make_attention_head(query)

        attention_output = self.attention_layer(query, key, value)
        enc_input = self.layer_norm_1(enc_input + attention_output)
        enc_output = self.feed_forward_layer(enc_input)
        return enc_output

# Initializing the model
self.attention_layer = MultiHeadAttention(d_input=512, d_model=512, num_heads=3)

    def make_attention_head(self, query):
        return (
            query,
            query if not self.config.use_residual_connection else query + self._compute_causal_mask_torch(query),
            query if not self.config.use_residual_connection else query + self._compute_causal_mask_torch(query),
        )

query = torch.randn(20, 32, 24)

# Inputs to the model
