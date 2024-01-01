
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, hidden_size):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.in_proj = torch.nn.Linear(hidden_size, num_attention_heads * 3)
        self.out_proj = torch.nn.Linear(num_attention_heads * 3, hidden_size)
 
    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, 3*self.hidden_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        # In the original code, `batch_size` is the second dimension
        batch_size = query.size()[0]
        qk = torch.matmul(query, key.transpose(-2, -1))
        scale_factor = (batch_size*1.0)/size(qk)[1]
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(1, 1024)

# Inputs to the model
query = torch.randn(1, 169, 1024)
key = torch.randn(1, 165, 1024)
value = torch.randn(1, 165, 1024)
