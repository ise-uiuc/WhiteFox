
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, __input__):
        qk = torch.matmul(__input__.query, __input__.key.transpose(-2, -1))
        scaled_qk = qk.div(1 / __input__.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=__input__.dropout_p)
        output = dropout_qk.matmul(__input__.value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_x = torch.randn(2, 4, 32)
input_y = torch.randn(2, 4, 32)
input = torch.nn.functional.linear(input_x, input_y)
input.query = input_x
input.key = input_y
input.value = input_y
input.scale_factor = 1 / 4
input.dropout_p = 0.5
