
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(3, 8, dropout=0.1)
 
    def forward(self, x1, x2, x3):
        qk = x2.transpose(-2, -1) @ x1
        v = torch.mean(x1 - x2 + x3, dim=1)
        scaled_qk = qk / 4
        softmax_qk = scaled_qk.softmax(dim=2)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_qk.transpose(-2, -1) @ v
        return output
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(4, 3, 128, 128)
x2 = torch.randn(4, 8, 128, 128)
x3 = torch.randn(4, 3, 128, 128)
