
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, x3, dropout_p):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = torch.sqrt(torch.tensor([x1.size(-1)]))
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16, 100)
x2 = torch.randn(1, 16, 200)
x3 = torch.randn(1, 400, 200)
dropout_p = torch.tensor([0.1])
