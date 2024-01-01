
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass
 
    def forward(self, __input__):
        qk = torch.matmul(__input_query_input__, __input_key_input__.transpose(-2, -1))
        scaled_qk = qk.mul(__input_scale__)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=__input_dropout_p__)
        output = dropout_qk.matmul(__input_value_input__)
        return output

# Initializing the model
m = Model()

# Inputs to the model
__input_query_input__ = torch.randn(1, 2, 100)
__input_key_input__ = torch.randn(1, 2, 100)
__input_scale__ = torch.randn(1)
__input_dropout_p__ = torch.randn(1)
__input_value_input__ = torch.randn(1, 2, 50)
