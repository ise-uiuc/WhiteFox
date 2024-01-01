 Inputs
- query: a tensor with shape [batch_size, head_num, num_query_vectors, dim_per_head]
- key: a tensor with shape [batch_size, head_num, num_keys, dim_per_head]
- value: a tensor with shape [batch_size, head_num, num_keys, dim_per_head]
- attn_mask: a tensor with shape [batch_size, num_query_vectors, num_keys] to be used in the attention computation. You will need to create this tensor based on the sequence lengths.

# Model
class Model(torch.nn.Module):
    def __init__(self, query, key, value, attn_mask, dropout_p=0.4, output_dim=128):
        super().__init__()
 
        # Store values
        num_query_vectors = list(query.size())[2]
        num_keys = list(key.size())[3]
 
        # Scaled Dot Product
        # Take a linear combination of the dot product of (query, key), and an attnetion mask
        self.dot_product = query @ key.transpose(-2, -1)
 
        # Normalize
        self.dot_product = self.dot_product / math.sqrt(query.size(-1))
 
        # Create the attention mask
        self.attn_mask = torch.zeros((num_query_vectors, num_keys)).to(key.device)
        for i, v in enumerate(attn_mask):
            self.attn_mask[:len(v), :len(v)] = attn_mask[i]
 
        # Compute attention weights
        self.attn_weight = self.dot_product + self.attn_mask
 
        # Softmax
        self.attn_weight = torch.softmax(self.attn_weight, dim=-1)
 
        # Dropout
        self.attn_weight = torch.nn.Dropout(dropout_p)(self.attn_weight)
 
        # Output
        self.output = self.attn_weight @ value
 
    def forward(self):
        return self.output

# Inputs to the model
query = torch.randn(1, 3, 12, 128)
key = <KEY>(1, 3, 24, 128)
value = torch.randn(1, 3, 24, 128)

# Attention mask that ignores the padding.
attn_mask = [
  [
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  ],
  [
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  ],
  [
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  ],
  [
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0],
  [1.0, 1.0, 1.0, 1.0]
  ],
]
