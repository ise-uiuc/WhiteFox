
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor1, input_tensor2):
        qk = torch.matmul(input_tensor1, input_tensor2.transpose(-2, -1))
        inv_scale_factor = 100
        dropout_p = 0.1
        dropout_qk = torch.nn.functional.dropout(qk.div(inv_scale_factor).softmax(dim=-1), p=dropout_p)
        output = dropout_qk.matmul(input_tensor2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor1 = torch.randn(400, 256)
input_tensor2 = torch.randn(256, 512)
