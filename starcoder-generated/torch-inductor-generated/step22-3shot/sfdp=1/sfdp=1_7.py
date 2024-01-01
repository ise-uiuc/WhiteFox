
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = dropout_p
 
    def forward(self, input_tensor1, input_tensor2, input_tensor3):
        qk = torch.matmul(input_tensor1, input_tensor2.transpose(-2, -1))
        inv_scale_factor = 1. / (input_tensor2.size(-1) ** 0.5)
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(input_tensor3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 20, 5)
x2 = torch.randn(1, 5, 20)
x3 = torch.randn(5, 20)
