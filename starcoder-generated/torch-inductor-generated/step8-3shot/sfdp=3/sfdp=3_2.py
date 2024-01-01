
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        q = x1.unsqueeze(dim=0)
        k = x2.unsqueeze(dim=0)
        v = x3.unsqueeze(dim=0)
        k = k.transpose(1, -1)
        scale_factor = 0.1
        dropout_p = 0.0
        qk = torch.matmul(q, k)
        scaled_qk = qk * scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)
        return output.view(x1.size())

# Initializing the model
m = Model()

# Inputs to the model
x1, x2, x3 = torch.randn(2, 3), torch.randn(2, 4), torch.randn(2, 4)
