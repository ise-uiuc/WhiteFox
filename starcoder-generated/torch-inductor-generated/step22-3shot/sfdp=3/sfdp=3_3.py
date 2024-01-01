
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, query, key, value, scale_factor, dropout_p):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1) 
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value) 
        return output

# Initializing the model
m = Model()
(query, key, value) = (torch.randn(1, 8, 64), torch.randn(1, 8, 64), torch.randn(1, 8, 64))
(scale_factor, dropout_p) = (torch.as_tensor(0.5), 0.2)
