
class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), "Embed dim must be divisible by num heads"
        self.scaling = self.head_dim**-0.5
        
        # Linear layers
        inner_dim = embed_dim
        
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias_=True, eps=1e-5, gain=1):

def linear(input, weight, bias):

class MLP(torch.nn.Module):
    def __init__(self, inner_dim, layer_norm_eps):
        super().__init__()
        self.norm1 = LayerNorm(inner_dim, layer_norm_eps) # Create a pointwise layer norm layer
        self.fc = Linear(inner_dim, inner_dim * 4, bias_=True, gain=np.sqrt(0.5)) # Create a pointwise linear layer with normalized coefficients

# Pointwise layer norm layer
class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):

def layer_norm(input, normalized_shape, weight, bias, eps):

# Pointwise feed forward network layer
class FFN(torch.nn.Module):
    def __init__(self, inner_dim, activation_fn, dropout):

class TransformerLayer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, activation_fn, dropout, attn_mask=None):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads) # Create a multi-head attention block
        self.norm1 = LayerNorm(embed_dim) # Create a pointwise layer norm layer
        self.ffn = FFN(ffn_dim, activation_fn, dropout) # Create a pointwise feedforward network
        self.norm2 = LayerNorm(embed_dim) # Create a pointwise layer norm layer
        self.dropout = torch.nn.Dropout(dropout) # Create a dropout layer
 
    def forward(self, x1):
        v1 = self.norm1(x1) # Applying layer norm to the input tensor
        v2 = self.self_attn(query=v1, key=v1, value=v1, attn_mask=attn_mask) # Applying Multi-head attention block to the input tensor
        v3 = v1 + v2 # Sum the two tensors
        v4 = self.norm2(v3) # Apply layer norm to the output tensor
        v5 = self.ffn(v4) # Apply FFN network to the tensor
        v6 = v4 + v5 # Sum the two tensors
        return self.dropout(v6) # Apply dropout with probability 0.1 to the output tensor

# Initializing the model
model = TransformerLayer(3, 1, 8, "gelu", 0.5)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
