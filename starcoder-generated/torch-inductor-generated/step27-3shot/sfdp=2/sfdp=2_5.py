
class Model(torch.nn.Module):
    def __init__(self, query_size, key_size, val_size, dropout_p=0.1):
        super().__init__()
        self.key = torch.nn.Linear(query_size, key_size, bias=False)
        self.value = torch.nn.Linear(query_size, val_size, bias=False)
        self.inv_scale_factor = torch.sqrt(torch.FloatTensor([key_size])).cuda()
        self.dropout_p = dropout_p
 
    def forward(self, query1, query2):
        k = self.key(query1)
        v = self.key(query2)
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_size=128, key_size=256, val_size=128)

# Input to the model
query1 = torch.randn(1, 128)
query2 = torch.randn(32, 128)
