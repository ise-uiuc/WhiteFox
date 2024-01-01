
class Model(torch.nn.Module):
    def __init__(self, d_k, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p
    
    def forward(self, query, key, inv_scale_factor, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(d_k=8, dropout_p=0.8)

# Inputs to the model
query = torch.randn(2, 3, 4, 8)
key = torch.randn(2, 3, 8, 8)
value = torch.randn(2, 3, 8, 8)
inv_scale_factor = torch.randn(1)
