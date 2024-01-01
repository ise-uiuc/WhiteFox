
class Model(torch.nn.Module):
    def __init__(self, d=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout_p=0.1, activation="relu", custom_encoder=None, custom_decoder=None):
        super().__init__()

    def forward(self, query, key, value, attention_mask=None, head_mask=None):
        nbatches = query.size(1)

        q = self.query(query).view(nbatches, -1, self.nhead, self.dhead).transpose(1, 2)
        k = self.key(key).view(nbatches, -1, self.nhead, self.dhead).transpose(1, 2)
        v = self.value(value).view(nbatches, -1, self.nhead, self.dhead).transpose(1, 2)

        q = q * self.scale
        attn_output, attn_output_weights = self.attend(q, k, v, attention_mask, head_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(nbatches, -1, self.nhead * self.dhead)
        attn_output = self.proj(attn_output)
        return attn_output, attn_output_weights

# Initializing the model
m = Model(10)

# Inputs to the model
query = torch.randn(4, 8, 10)
key = torch.randn(4, 8, 10)
value = torch.randn(4, 8, 10)
