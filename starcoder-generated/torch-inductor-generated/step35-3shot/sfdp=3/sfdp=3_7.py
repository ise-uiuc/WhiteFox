
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = torch.sqrt(torch.tensor(x1.size(-1)))
        scaled_qk = scale_factor * qk
        softmax_qk = torch.softmax(scaled_qk, dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 35, 35)
x2 = torch.randn(1, 128, 35, 35)
