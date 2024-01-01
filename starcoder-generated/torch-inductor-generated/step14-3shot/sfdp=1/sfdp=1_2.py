
class Model(torch.nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p # Save the dropout probability
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        inv_scale_factor = math.sqrt(x1.size(-1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(x3)
        return output

# Initializing the model
m = Model(dropout_p=0.3)

# Inputs to the model
x1 = torch.randn(2, 512, 768)
x2 = torch.randn(256, 512, 768)
x3 = torch.randn(256, 768, 512)
