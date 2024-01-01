
class Model(torch.nn.Module):
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = torch.tensor([64 * 64]).float() # [1]
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
    
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(batch_size, num_heads, key_length, key_length)
key = torch.randn(batch_size, num_heads, key_length, key_length)
value = torch.randn(batch_size, num_heads, key_length, key_length)
