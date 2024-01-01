
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2):
        qk = torch.matmul(self.query, self.key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(self.value)
        return output
    
# Initializing the model
m = Model(query=torch.randn(16, 64, 16), key=torch.randn(16, 64, 20), value=torch.randn(16, 64, 20), inv_scale_factor=math.sqrt(64), dropout_p=0.5)

# Inputs to the model
x1 = torch.randn(2, 8, 64, 16)
x2 = torch.randn(2, 8, 64, 20)
