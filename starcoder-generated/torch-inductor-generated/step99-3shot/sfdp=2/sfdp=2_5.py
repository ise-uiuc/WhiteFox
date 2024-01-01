
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, qk, inv_scale_factor, dropout_p, value):
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
qk = torch.randn(8, 64, 256, 64)
inv_scale_factor = torch.tensor([8])
dropout_p = 0.3
value = torch.randn(8, 256, 64)
