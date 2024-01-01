
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input, key, value, mask=None):
        inv_scale_factor = torch.tensor(8.0).log_inverse()
        qk = torch.matmul(input, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.5)
        output = dropout_qk.matmul(value)
        if mask is not None:
            output = output * (mask.unsqueeze(-1))
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
input = torch.randn(32, 2, 512)
key = torch.randn(32, 512, 128)
value = torch.randn(32, 512, 128)
mask = torch.randint(0, 2, (32, 1, 128))
