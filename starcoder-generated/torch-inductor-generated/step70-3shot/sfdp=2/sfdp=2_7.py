
class Model(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w_q = torch.nn.Linear(hidden_size * hidden_size, hidden_size * hidden_size)
        self.w_k = torch.nn.Linear(hidden_size * hidden_size, hidden_size * hidden_size)
        self.w_v = torch.nn.Linear(hidden_size * hidden_size, hidden_size * hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.w_o = torch.nn.Linear(hidden_size * hidden_size, hidden_size * hidden_size)
    
    def forward(self, query, key, value, inv_scale_factor):
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)
        
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        
        output = dropout_qk.matmul(v)
        return self.w_o(output)

# Initializing the model
# For simplicity set hidden_size as 123
m = Model(123)

# Inputs to the model
query = torch.randn(4, 123, 123)
key = torch.randn(4, 123, 123)
value = torch.randn(4, 123, 123)
inv_scale_factor = torch.randn(4, 123)
