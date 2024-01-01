
class Model(torch.nn.Module):
    def __init__(self, p_attention_dropout, scale_factor):
        super().__init__()
        self.dropout = torch.nn.Dropout(p_attention_dropout)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.scale_factor = scale_factor
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factror)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
p_attention_dropout = 0.5
scale_factor = 1 / sqrt(query_size)
m = Model(p_attention_dropout, scale_factor)

# Inputs to the model
query = torch.randn(1, query_size, v_length)
key = torch.randn(1, query_size, k_length)
value = torch.randn(1, query_size, v_length)
