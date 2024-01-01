
class AttentionLayer(torch.nn.Module):
    def __init__(self, query_size, key_size, attention_size, dropout_p=0):
        super().__init__()
        self.attention_size = attention_size
   
        self.scale_factor = 1.0 / math.sqrt(attention_size)
        self.dropout_p = dropout_p
 
        self.query_projection = torch.nn.Linear(query_size, attention_size)
        self.key_projection = torch.nn.Linear(key_size, attention_size)
 
    def forward(self, x1):
        q = self.query_projection(x1)
        k = self.key_projection(x1)
 
        q_transpose_k = torch.matmul(q, k.transpose(-2, -1))
 
        scaled_qk1 = q_transpose_k.mul(self.scale_factor)
        softmax_qk1 = scaled_qk1.softmax(dim=-1)
        dropout_qk1 = torch.nn.functional.dropout(softmax_qk1, p=self.dropout_p)
        output = torch.matmul(dropout_qk1, x1)
        return output

# Initializing the model
m = AttentionLayer(3, 4, 8, 0.5)

# Inputs to the model, note that the query and the key should have the same shape
x1 = torch.randn(1, 3, 64, 64)
