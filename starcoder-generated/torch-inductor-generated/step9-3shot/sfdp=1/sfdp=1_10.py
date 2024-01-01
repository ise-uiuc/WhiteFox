
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, inv_scale_factor, dropout_p, attention_mask):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        y = dropout_qk.matmul(value)
        return y
 
# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 64)
key = torch.randn(1, 16, 64)
value = torch.randn(1, 16, 64)
inv_scale_factor = torch.randn(1, 16)
dropout_p = torch.empty(1).uniform_(0, 1)
attention_mask = torch.empty(1, 16, 64).bernoulli_(0)
