
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, inv_scale_factor, dropout_p):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = F.softmax(scaled_qk, dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = F.dropout(softmax_qk, p=dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(v) # Compute the dot product of the dropout output and the value tensor
        return output

# Initializing the model
dropout_p = 0.5
m = Model()

# Input to the model
q = torch.randn(1, 2, 8)
k = torch.randn(1, 4, 8)
v = torch.randn(1, 4, 8)
inv_scale_factor = 10
