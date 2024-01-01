
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, k.transpose(-2, -1))
        scaled_qk = qk.div(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
# Initializing the model
m = Model()

# Input for the model
_input_1_1 = torch.randn(1, 256, 5, 300)
_input_1_2 = torch.randn(1, 256, 5, 300)
_input_2_1 = torch.randn(1, 256, 5, 300)
__input_3 = pow(5, -(1./2)) 
__input_4 = 0.1
