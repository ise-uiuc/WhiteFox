
def forward(self, query, key, value, mask, dropout_p=0.5):
    inv_scale_factor = 1.0 / math.sqrt(query.size(-1)) # Compute the inverse of the scale factor
    qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
    scaled_qk = qk.div(inv_scale_factor) # Scale the dot product by the inverse scale factor
    softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
    dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
    output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
    output = self.attn_output_projection(output).masked_fill_(mask.unsqueeze(-1), 0.0).contiguous() # Apply output projection to each sub-array of the output
    return output

# Initializing the model
m = Model()

# Inputs to the model
batch_size = 1
seq_length = 256
dim = 128
query = torch.randn(batch_size, seq_length, dim)
key = torch.randn(batch_size, seq_length, dim)
value = torch.randn(batch_size, seq_length, dim)
mask = torch.logical_or(torch.arange(seq_length).to(query.device) > 250, torch.arange(seq_length).to(query.device) < 100)
dropout_p = 0.2
