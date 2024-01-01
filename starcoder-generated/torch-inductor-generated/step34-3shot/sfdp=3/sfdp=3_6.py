
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_i, v_j, dropout_p=0):
        qk = torch.matmul(v_i, v_j.transpose(-2, -1))
        scale_factor = 1.0 / math.sqrt(v_i.size(-1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v_j)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v_i = torch.randn(1, 3, 64, 64)
v_j = torch.randn(1, 3, 64, 64)
