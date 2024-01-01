
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout_p = 0.1
 
    def forward(self, query, key, value):
        scale_factor = (key.shape[-1] / query.shape[-1]).pow(0.25)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = F.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 8, 48)
key = torch.randn(1, 8, 96)
value = torch.randn(1, 8, 96)
model_output = m(query, key, value)

