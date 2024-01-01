
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / (sqrt(dk))
 
    def forward(self, q, k, v, dropout_p):
        qk = q @ k.transpose(-2, -1)
        scaled_qk = qk * self.scale_factor
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk @ v
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 16, 128)
key = torch.randn(1, 64, 128)
value = torch.randn(1, 128, 128)
