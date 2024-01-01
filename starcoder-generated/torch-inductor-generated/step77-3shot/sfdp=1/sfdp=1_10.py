
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, __input_value__):
        qk = torch.matmul(__input_value__, __input_value__.transpose(__key_dim__, __value_dim__))
        scaled_qk = qk.div(__input_value__.size(__key_dim__) ** -0.25)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=__dropout_p__)
        output = dropout_qk.matmul(__input_value__)
        return output

# Initializing the model
__model__ = Model()

# Inputs to the model
__input_tensor_type__ = torch.FloatTensor
__input_dim__ = (1, 64, 8)
__key_dim__ = 1
__value_dim__ = 2
__dropout_p__ = 0.1
__scale_factor__ = __input_tensor_type__(1).fill_(1 / __input_dim__[__key_dim__])
x1 = __input_tensor_type__(*__input_dim__)
