
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, dropout_p=0.6, inv_scale_factor=1/(10**6)):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1, dtype=torch.float32)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p, training=True)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
