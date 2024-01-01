
class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
 
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
 
        self.query_projection = torch.nn.Conv2d(d_model, d_model, 1)
        self.key_projection = torch.nn.Conv2d(d_model, d_model, 1)
        self.value_projection = torch.nn.Conv2d(d_model, d_model, 1)
 
 
    def compute_qkv(self, x):
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
 
        h, w = key.shape[-2:]
        h = h // self.num_heads
        w = w // self.num_heads
 
        query = torch.reshape(query, shape=(-1, self.num_heads, self.d_head, h, w))
        key = torch.reshape(key, shape=(-1, self.num_heads, self.d_head, h, w))
        value = torch.reshape(value, shape=(-1, self.num_heads, self.d_head, h, w))
 
        return query, key, value
 
 
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.layer = MultiHeadAttentionLayer(d_model, num_heads)
        self.scale_factor = d_model ** 0.5
 
    def forward(self, x, dropout_p=0.0):
        query, key, value = self.layer.compute_qkv(x)
 
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = 1.0 / self.scale_factor
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
 
        return output
 
 
class Model(torch.nn.Module):
    def __init__(self, channel_axis, d_model, num_heads, dropout_p=0.0):
        super().__init__()
 
        self.projection = torch.nn.Conv2d(2**channel_axis, d_model, 1)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
 
    def forward(self, x):
        x = self.projection(x)
        x = self.multi_head_attention(x)
        return x

# Initializing the model
