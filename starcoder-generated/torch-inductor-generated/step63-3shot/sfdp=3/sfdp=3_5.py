
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads,
                 scale_factor, dropout_p):
        super().__init__()
        self.multihead_attention = torch.nn.MultiheadAttention(
            input_size, num_heads, dropout=dropout_p)
        self.query_linear = torch.nn.Linear(input_size, hidden_size)
        self.value_linear = torch.nn.Linear(input_size, hidden_size)
        self.output_linear = torch.nn.Linear(hidden_size, input_size)
        self.embedding = torch.nn.Embedding(num_layers, hidden_size)
        self.scale_factor = torch.nn.Parameter(
            scale_factor, requires_grad=True)

    def forward(self, query, key, value, mask):
        q = self.query_linear(query)
        v = self.value_linear(value)
        k = (
            key if key is not None
            else value)
        mask = (
            mask if mask is not None
            else torch.ones(
                query.shape[0], query.shape[0], **val_kwargs))
        embeddings = self.embedding.weight.repeat(
            q.shape[0], 1, 1)
        scaled_attn, _ = self.multihead_attention(
            q, k, v, attn_mask=mask, key_padding_mask=None, 
            need_weights=False, attn_bias_k=None, attn_bias_v=None)
        out = self.output_linear(scaled_attn)
        return out * self.scale_factor

# Initializing the model
input_size = 8
hidden_size = 4
num_layers = 2
num_heads = 2
scale_factor = torch.rand(1)
dropout_p = 0.2
m = Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
          scale_factor=scale_factor, dropout_p=dropout_p)
print("Model generated!")
# Inputs to the model
query = torch.randn(1, 1, input_size)
key = torch.randn(1, 1, input_size)
value = torch.randn(1, 1, input_size)
mask = torch.ones(1, 1).bool()

m.eval()
with torch.no_grad():
    out = m(query, key, value, mask)
