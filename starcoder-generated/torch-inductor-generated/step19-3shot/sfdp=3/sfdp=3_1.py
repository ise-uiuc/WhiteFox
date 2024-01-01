
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor(0.5))
 
    def forward(self, value, query, key, dropout_p):
        softmax_qk = torch.nn.functional.softmax(query @ key.transpose(-2, -1), dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        return dropout_qk @ value

# Initializing the model
m = Model()

# Inputs to the model
value = torch.randn(1, 2, 3)
query = torch.randn(1, 3, 4)
key = torch.randn(1, 4, 2)
dropout_p = torch.nn.Parameter(torch.tensor(0.7))
