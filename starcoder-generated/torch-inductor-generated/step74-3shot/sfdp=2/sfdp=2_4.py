
class Model(torch.nn.Module):
    def __init__(self, hidden_dim, num_heads, scale_factor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.linear_q = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_o = torch.nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, hidden_state, attention_mask, dropout_p):
        query = self.linear_q(hidden_state)
        key = self.linear_k(hidden_state)
        value = self.linear_v(hidden_state)
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.scale_factor) # Scale the dot product by the scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the dot product of the dropout output and the value
        return output

# Initializing the model
m = Model(hidden_dim=16, num_heads=4, scale_factor=1/16)

# Inputs to the model
hidden_state = torch.randn(1, 16)
attention_mask = torch.randn(1, 1, 1, 16)
dropout_p = 0.5
