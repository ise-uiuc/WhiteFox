
class Model(torch.nn.Module):
    def __init__(self, hidden_size=42, query_size=42, key_size=42, value_size=42, output_size=42, dropout_p=0.0, inv_scale_factor=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.dropout_p = dropout_p
 
    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_qk = qk.div(self.inv_scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p)
        output = dropout_qk.matmul(value)
        return output

# Initializing the model
m = Model(query_size=input_feature_dim, key_size=input_feature_dim, value_size=output_channel_size)

# Inputs to the model
query = torch.randn(4, 4, input_feature_dim)
key = torch.randn(8, 2, input_feature_dim)
value = torch.randn(8, 2, output_channel_size)
