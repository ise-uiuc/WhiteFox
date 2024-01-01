
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1/(d_k**-0.5)
 
    def forward(self, query, key, value, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1/(d_k**-0.5)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(dropout_p)
 
    def forward(self, query, key, value, dropout_p=0.1):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.scale_factor)
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m1 = Model1()
m2 = Model2()

# Inputs to the model
query = torch.randn(batch_size, seq_length, d_model)
key = torch.randn(batch_size, seq_length, d_model)
value = torch.randn(batch_size, seq_length, d_model)
