
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self,
                query,
                key,
                value,
                query_mask = None,
                key_mask = None,
                inv_scale_factor = 1.,
                dropout_p = 0.6):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the parameters of the model, including the inverse scale factor, the dropout probability, and the mask tensors
# Input and output tensors should have same size
# Output mask tensor should have the same size as the output tensor
batch = 2
query_seq_len = 3
key_seq_len = 5
n_head = 3
n_attn_units = 2
query = torch.randn(batch, n_head, query_seq_len, n_attn_units)
key = torch.randn(batch, n_head, key_seq_len, n_attn_units)
value = key
query_mask = torch.FloatTensor([[1,1,0],[0,1,1]])
inv_scale_factor = 1 / np.sqrt(n_attn_units) # inverse scale factor is calculated because the dot product of different dimension vectors should not be scaled
dropout_p = 0.6

# Inputs to the model
