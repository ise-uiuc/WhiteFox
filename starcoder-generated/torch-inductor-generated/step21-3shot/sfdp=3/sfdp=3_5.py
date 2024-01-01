
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
        self.query_conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.key_conv = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
 
        self.scale_factor = 1 / math.sqrt(64)
 
        self.dropout_p = 0.5
 
    def forward(self, x1):
        q = self.query_conv(x1)
        k = self.key_conv(x1)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        v = torch.randn_like(k)
        output = dropout_qk.matmul(v)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 64, 24, 24)
