
class Model(torch.nn.Module):
    def __init__(self, scale_factor=1.0, dropout_p=0.5):
        super().__init__()
        self.scale_factor = scale_factor
        self.dropout_p = dropout_p
 
 
    def forward(self, q, k, v):
        qk_dots = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk_dots * self.scale_factor
        softmax_qk = scaled_QK.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(v)
        return output


# Initializing the model
m = Model()
 
# Inputs to the model
query = torch.randn(1, 4, 32, 56)
key = torch.randn(1, 4, 56, 32)
value = torch.randn(1, 4, 56, 56)
