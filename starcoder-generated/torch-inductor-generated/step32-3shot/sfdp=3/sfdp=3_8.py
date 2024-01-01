
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
  assert len(query.shape) == len(key.shape) == len(value.shape) == 3, \
  "Inputs: query, key, value should all be 3-D tensors"
  dims = query.shape[-1]
  if mask is not None:
    mask = mask.unsqueeze(1).expand(mask.size(0), dims, mask.size(1))

  # Compute the dot product of the query and the key tensors
  scaled_qk = torch.matmul(query, key.transpose(-2, -1))

  # Scale the dot product by a factor of square root of the dimensionality
  if mask is not None:
    assert mask.size(1) == scaled_qk.size(-1) == scaled_qk.size(-2)
    scaled_qk = scale_by_square_root_dim(scaled_qk, mask.unsqueeze(-1))
    scaled_qk.masked_fill_(mask == 0, -1e9)
  else:
    scaled_qk = scaled_qk / math.sqrt(query.size(-1))

  # Apply softmax to the scaled dot product
  softmax_qk = scaled_qk.softmax(-1)

  # Apply dropout
  if dropout is not None:
    softmax_qk = dropout(softmax_qk)

  # Compute the dot product of the dropout output and the value tensor
  output = torch.matmul(softmax_qk, value)

  return output

class MultiHeadAttention(torch.nn.MultiHeadAttention):
    def forward(self, query, key, value, attn_mask=None):
        return scaled_dot_product_attention(query, key, value, attn_mask)

class TransformerBlock(nn.Module):
  def __init__(self):
    super().__init__()

    self.norm_1 = norm_layer
    self.attn_prenorm_1 = partial(self.attn_prenorm, dropout=None)

    self.attn_1 = self.attn_prenorm_1()

    self.norm_2 = norm_layer
    self.mlp_prenorm_1 = partial(self.mlp_prenorm, dropout=None)
    self.mlp_prenorm_2 = partial(self.mlp_prenorm, dropout=None)

    self.mlp_1 = self.mlp_prenorm_1()
    self.mlp_2 = self.mlp_prenorm_2()

  def forward(self, x):

    input_tensor = x

    # Apply Layer Normalization to the input tensor
    norm_1_output = self.norm_1(input_tensor)

    # Apply pre-normalization in self-attention
    attn_prenorm_1_output = self.attn_prenorm_1(norm_1_output)

    # Apply self-attention
    attn_1_output = self.attn_1(norm_1_output, attn_prenorm_1_output)

    # Apply pre-normalization in the fully-connected part of the residual connection
    residual_attn_1 = attn_1_output + input_tensor

    # Apply Layer Normalization to the input tensor
    norm_2_output = self.norm_1(residual_attn_1)

    # Apply pre-normalization in the residual connection with the fully-connected part
    residual_attn_2 = norm_2_output + input_tensor

    return input_tensor

class TransformerBlock(nn.Module):
  def __init__(self):
    super().__init__()

    self.norm_1 = norm_layer
    self.attn_prenorm_1 = partial(self.attn_prenorm, dropout=None)

    self.mlp_prenorm_1 = partial(self.mlp_prenorm, dropout=None)
    self.mlp_prenorm_2 = partial(self.mlp_prenorm, dropout=None)

    self.mlp_1 = self.mlp_prenorm_1()
    self.mlp_2 = self.mlp_prenorm_2()

  def forward(self, x):

    input_tensor = x

    # Apply Layer Normalization to the input tensor
    norm_1_output = self.norm_1(input_tensor)

    # Apply pre-normalization in self-attention
    attn_prenorm_1_output = self.attn_prenorm_1(norm_1_output)

    # Apply pre-normalization in the fully-connected part of the residual connection
    residual_attn_1 = attn_prenorm_1_output + input_tensor

    # Apply Layer Normalization to the input tensor
    norm_2_output = self.norm_1(residual_attn_1)

    # Apply pre-normalization in the residual connection with the fully-connected part
    residual_attn_2 = norm_2_output + input_tensor

    return input_tensor

# Initializing the model
m = MultiHeadAttention()

# Inputs to the model
x1, x2, x3 = (torch.randn(8, 64, 1024), torch.randn(8, 64, 2048)
           torch.randn(10, 64, 1024))
z = m(x2, x3, x1)

