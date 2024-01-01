
class Model(torch.nn.Module):
    def __init__(self, d_model, d_qk, scale_factor, dropout_p):
        super().__init__()
        self.qk = torch.nn.Linear(d_model, d_qk)
        self.scaled_qk = torch.nn.Linear(d_qk, 1)
        self.inv_scale_factor = 1 / scale_factor
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = self.qk(query)
        scaled_qk = self.scaled_qk(qk).div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(128, 32, 10, 0.5)

# Inputs to the model
query = torch.randn(5, 128)
key = torch.randn(6, 128)
value = torch.randn(6, 128)
