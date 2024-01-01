
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor1, input_tensor2, inv_scale_factor, dropout_p):
        qk = torch.matmul(input_tensor1, input_tensor2.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(input_tensor1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 100, 50)
x2 = torch.randn(1, 3, 100, 50)
inv_scale_factor = torch.rand(1)
dropout_p = torch.rand(1)
