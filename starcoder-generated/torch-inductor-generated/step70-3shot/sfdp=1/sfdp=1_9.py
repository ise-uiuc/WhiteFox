
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = torch.nn.Linear(hidden_state, 1)
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model for testing purposes
query = torch.randn(10, 16, hidden_state)
key = torch.randn(10, 20, hidden_state)
value = torch.randn(10, 20, hidden_state)
inv_scale_factor = torch.randn(1)
dropout_p = torch.randn(1)
