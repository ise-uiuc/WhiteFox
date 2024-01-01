
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, __input__):
        qk = torch.matmul(__input__.q, __input__.k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(__input__.v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
__input__ = Input(q, k, v, inv_scale_factor, dropout_p)
