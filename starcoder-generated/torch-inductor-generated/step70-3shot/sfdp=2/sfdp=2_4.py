
class Model(torch.nn.Module):
   def __init__(self):
       super().__init__()
       self.dropout = torch.nn.Dropout(dropout_p)
 
       self.query = torch.nn.Linear(hidden_size, hidden_size)
       self.key = torch.nn.Linear(hidden_size, hidden_size)
       self.value = torch.nn.Linear(hidden_size, hidden_size)
 
   def forward(self, query, key, value):
       scaled_qk = torch.matmul(query, self.key.transpose(-2, -1)).div(inv_scale_factor)
       softmax_qk = scaled_qk.softmax(dim=-1)
       dropout_qk = self.dropout(softmax_qk)
       output = torch.matmul(dropout_qk, self.value)
       return output

# Inputs to the model
query = torch.randn(1, hidden_size, seq_length)
key = torch.randn(1, hidden_size, seq_length)
value = torch.randn(1, hidden_size, seq_length)
