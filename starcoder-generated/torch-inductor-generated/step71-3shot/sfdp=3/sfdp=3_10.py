
class MultiHeadAttention(nn.Module):
    __constants__ = ['num_heads', 'head_dim', 'dropout_p']
 
    def __init__(self, input_dim, embed_dim: int, num_heads: int = 4, dropout_p: float = 0, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
 
    def forward(self, q, k, v):
        B, N, C = q.shape # B: batch size, N: sequence length, C: channel size
        qkv_same = q.data_ptr() == k.data_ptr() == v.data_ptr() # True if the three tensors point to the same underlying storage
        kv_same = k.data_ptr() == v.data_ptr() # True if the two tensors point to the same underlying storage
        qkv_size = (B, N, self.num_heads, C // self.num_heads) # The size of the query, key, and value tensors
        flat_qkv_size = (B * N, self.num_heads, C // self.num_heads) # The size of the flatten query, key, and value tensors
        flat_q = q.view(flat_qkv_size) # Reshaping the query tensor
        flat_k = v if kv_same else k.view(flat_qkv_size) # Reshaping the key tensor
        flat_v = q if qkv_same else v.view(flat_qkv_size) # Reshaping the value tensor
        scale_factor = 1 / math.sqrt(self.head_dim) # Scale factor used to scale the dot product computed by the attention layer
        scale_q = q * scale_factor
        scale_k = k * scale_factor
        # Performing the self-attention operation
        flat_weighted_attn = torch.matmul(scale_q, scale_k.transpose(-2, -1)) # The dot product of the scaled query and transposed scaled key
        # Applying the softmax function and dropout to the scaled dot product between query and key
        softmax_attn = F.softmax(flat_weighted_attn, dim=-1)
        dropout_attn = F.dropout(softmax_attn, p=self.dropout_p)
        flat_output = torch.matmul(dropout_attn, flat_v) # The dot product of the dropout result and the reshaped value
        # Returning the output tensor after reshaping it
        output = flat_output.view(qkv_size)
        return output
 
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_p: float = 0):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, embed_dim, num_heads, dropout_p)
 
# Instantiating the model
m = MultiHeadAttentionLayer(256, 8, 0.1)

# Inputs to the model
q = torch.randn(20, 256, 128)
k = torch.randn(20, 256, 128)
v = torch.randn(20, 256, 128)
