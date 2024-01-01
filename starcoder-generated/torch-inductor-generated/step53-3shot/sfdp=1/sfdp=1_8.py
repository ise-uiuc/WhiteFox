
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def softmax_attention(self, query, key, inv_scale_factor=None, dropout_p=0.0):
        qk = torch.matmul(query, key.transpose(-2, -1))
        if inv_scale_factor is not None:
            qk = qk.div(inv_scale_factor)
        softmax_qk = qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk
    
    def forward(self, query, key, value, inv_scale_factor=None, dropout_p=0.0):
        qk = self.softmax_attention(query, key, inv_scale_factor=inv_scale_factor, dropout_p=dropout_p)
        output = qk.matmul(value)
        return output

# Initializing the model with the specified keyword arguments
m = Model(dropout_p=0.2)

# Inputs to the model
query = torch.randn(2, 4, 1, 16)
key = torch.randn(2, 4, 2, 16)
value = torch.randn(2, 4, 2, 16)
