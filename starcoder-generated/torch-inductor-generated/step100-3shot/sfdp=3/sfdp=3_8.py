
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self,query, key, value, scale_factor, dropout_p=0.3):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_dot_product_attention = qk * scale_factor
        softmax_qk = softmax(scaled_dot_product_attention, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 20, 256)
key=torch.randn(1, 10, 20)
value = torch.randn(1, 10, 30)
__scale_factor__ = torch.Tensor([0.1])
