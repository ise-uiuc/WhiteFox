
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor, key, value, inv_scale_factor, dropout_p):
        result_1 = torch.matmul(query, key.transpose(-2, -1))
        result_2 = result_1.div(inv_scale_factor)
        result_3 = result_2.softmax(dim=-1)
        result_4 = torch.nn.functional.dropout(result_3, p=dropout_p)
        result_5 = result_4.matmul(value)
        return result_5

# Initializing the model
m = Model()

# Inputs to the model
dropout_p = 0.5
query = torch.randn(2, 3, 48, 64)
key = torch.randn(2, 3, 432, 56)
value = torch.randn(2, 3, 48, 56)
inv_scale_factor = torch.zeros((2, 3, 1, 1)).fill_(10).requires_grad_()
