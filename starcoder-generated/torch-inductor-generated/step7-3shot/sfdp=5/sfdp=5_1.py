
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, input_tensor, input_mask, key, value):
        qk = input_tensor @ key.transpose(-2, -1)
        qk = qk + input_mask
        attn_weight = torch.softmax(qk, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, True)
        output = attn_weight @ value
        return output

# Initializing the model
m = Model()

# Inputs to the model
input_tensor = torch.randn(1, sequence_length, hidden_size)
input_mask = torch.zeros(1, sequence_length, sequence_length)
key = torch.randn(1, sequence_length, hidden_size)
value = torch.randn(1, sequence_length, hidden_size)

