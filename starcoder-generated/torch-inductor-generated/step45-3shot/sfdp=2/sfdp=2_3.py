
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, v, d):
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
# Initializing the model
m = Model()

# Inputs to the model
d = 128
q = torch.randn(1, d, 512)
k = torch.randn(1, d, 495)
v = torch.randn(1, d, 495)
