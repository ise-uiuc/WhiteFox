
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_1 = torch.nn.Dropout(dropout_p)
    
    def forward(self, x1, x2):
        x11 = self.dropout_1(x1)
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        t2 = dropout_qk.matmul(x2)
        return t2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(25, emb_dim)
x2 = torch.randn(25, emb_dim)
