
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p, input_mask=None):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        if input_mask is not None:
            softmax_qk = softmax_qk.masked_fill(input_mask.unsqueeze(-1), float('-inf'))
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(2, 8, 512)
key = torch.randn(2, 8, 512)
value = torch.randn(2, 8, 512)
scale_factor = 0.1
dropout_p = 0.5
input_mask = torch.Tensor([[True, False, False], [True, True, False]])
