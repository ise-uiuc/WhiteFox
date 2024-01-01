
class Model(torch.nn.Module):
    def __init__(self, num_heads=1, dropout_p=0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
    def forward(self, queries, keys, values, mask=None):
        _, q_seq_len, _, _ = queries.shape
        _, k_seq_len, _, _ = keys.shape
        _, _, d_model, _ = values.shape
        num_heads = self.num_heads
        dropout_p = self.dropout_p
 
        q = queries
        k = keys
        v = values
 
        scaled_q = q
        scaled_k = k.transpose(-2, -1)
        scaled_q = scaled_q.reshape(q_seq_len, -1, num_heads, d_model // num_heads).transpose(0, 1)
        scaled_k = scaled_k.reshape(k_seq_len, -1, num_heads, d_model // num_heads).transpose(0, 1)
        qk = torch.matmul(scaled_q, scaled_k)
 
        scale_factor = torch.tensor(d_model // num_heads).sqrt().type_as(qk)
        inv_scale_factor = torch.tensor(1 / d_model // num_heads).sqrt().type_as(qk)
        
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v.transpose(-2, -1))
 
        return output, dropout_qk, softmax_qk, q, k, v

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 20, 512, 64)
key = torch.randn(2, 40, 512, 64)
value = torch.randn(2, 40, 512, 64)
if mask is not None:
	mask = torch.randn(20, 40).type(torch.bool)
