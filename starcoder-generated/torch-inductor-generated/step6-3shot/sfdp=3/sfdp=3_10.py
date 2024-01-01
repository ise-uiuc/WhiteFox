
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.tensor(1000, dtype=torch.float) # Set the scale factor
        self.dropout_p = torch.tensor(0.2, dtype=torch.float) # Set the dropout rate

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
query = torch.randn(1, 10, 25)
key = torch.randn(1, 10, 25)
value = torch.randn(1, 10, 50)
