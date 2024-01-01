
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, inv_scale_factor, dropout_p, dropout_enabled):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        if dropout_enabled:
            dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        else:
            dropout_qk = softmax_qk
        return dropout_qk.matmul(value)

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(3, 256, 20)
key = torch.randn(3, 256, 256)
value = torch.randn(3, 256, 256)
