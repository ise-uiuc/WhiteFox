
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor1, input_tensor2, input_tensor3, input_tensor4):
        qk = torch.matmul(input_tensor1, input_tensor2.transpose(-2, -1))
        scaled_qk = qk.mul(0.4)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.4)
        output = dropout_qk.matmul(input_tensor3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor1 = torch.randn(1, 10, 5)
input_tensor2 = torch.randn(1, 20, 10)
input_tensor3 = torch.randn(1, 20, 64)
input_tensor4 = torch.randn(1, 64, 4)
