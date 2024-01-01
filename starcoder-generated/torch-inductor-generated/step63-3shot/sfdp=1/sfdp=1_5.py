
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.rand(1, 4)
        self.key = torch.rand(2, 4)
        self.value = torch.rand(2, 4)
        self.scale_factor = torch.rand(1)
        self.dropout_p = torch.rand(1)
 
    def forward(self):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model()

# Inputs to the model
