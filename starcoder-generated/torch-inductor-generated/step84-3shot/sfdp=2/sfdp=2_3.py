
class Model(torch.nn.Module):
    def __init__(self, args, hidden_size):
        super().__init__()
        self.query_proj = torch.nn.Linear(hidden_size, hidden_size)
        self.key_proj = torch.nn.Linear(hidden_size, hidden_size)
 
    def forward(self, query, key):
        q_new = self.query_proj(query)
        k_new = self.key_proj(key)
        qk = q_new.matmul(k_new.transpose(-1, -2))
        scaled_qk = qk.div(args.scale)
        dropout_qk = torch.dropout(softmax_qk, p=args.dropout)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
hidden_size = 2048
m = Model(args, hidden_size)

# Inputs to the model
query = torch.randn(1, hidden_size)
