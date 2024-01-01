
class MultiheadAttention(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout_p):
        super().__init__()
        self.query_scale = d_model ** -0.5
        self.qk_net = torch.nn.Linear(d_model, d_model * 2)
        self.v_net = torch.nn.Linear(d_model, d_model)
        self.output_layer = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(p=dropout_p, inplace=True)
        self.num_heads = num_heads

    def forward(self, query, key, value, attention_mask=None):
        qk = self.qk_net(query)
        qk = qk.reshape(qk.shape[0], qk.shape[1], 2, self.num_heads, -1)
        qk = qk.permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1] # Separate into heads. (num_heads, batch_size, num_objects, key_dimensions)

        key = key.transpose(-2, -1) # (batch_size, key_dimensions, num_objects)
        dot_product = q @ k * self.query_scale
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) # (batch_size, 1, num_objects, key_dimensions)
            dot_product = dot_product + attention_mask
        attention_weights = torch.softmax(dot_product, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = (value @ attention_weights.transpose(-2, -1)).join_batch_dims(dim=0) # (batch_size, num_objects, output_dimensions)
        output = self.output_layer(output)
        return output

# Initializing the model
d_model = 128
m = MultiheadAttention(num_heads=8, d_model=d_model, dropout_p=0.0)

# Inputs to the model
query = torch.randn(1, 32, d_model)
key = torch.randn(1, 64, d_model)
value = torch.randn(1, 64, d_model)
