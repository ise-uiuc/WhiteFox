
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, __input__):
        scale_factor = 0.225
        query = __input__[__mask__(0, 1, 2)].cuda()
        key = __input__[__mask__(0, 2, 1)].cuda()
        value = __input__.cuda()
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.75)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()


# Inputs to the model
x = torch.randn(32, 4, 5)
