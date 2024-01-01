
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, v1, v2, v3, v4):
        qk = torch.matmul(v1, v2.transpose(-2, -1))
        scaled_qk = qk.div(v3)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, v4)
        output = dropout_qk.matmul(v2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(1, 768, 6, 6)
v2 = torch.randn(1, 768, 4, 4)
v3 = torch.Tensor([0.1])
v4 = torch.Tensor([0.0])
