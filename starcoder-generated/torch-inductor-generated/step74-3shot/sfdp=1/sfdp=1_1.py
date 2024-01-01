
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, dropout_p):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.wq = torch.nn.Linear(in_channels, out_channels, bias=True)
        self.wk = torch.nn.Linear(in_channels, out_channels, bias=True)
        self.wv = torch.nn.Linear(in_channels, out_channels, bias=True)
 
    def forward(self, query, key, value):
        q = self.wq(query) # Apply the linear transformation to the query
        k = self.wk(key) # Apply the linear transformation to the key
        v = self.wv(value) # Apply the linear transformation to the value
        q, k, v = self._reshape_inputs(q, k, v)
        dropout_qk = self._scaled_dot_product_attention(query, key, value)
        output = self._final_linear_projection(dropout_qk)
 
        return output
 
    def _reshape_inputs(self, query, key, value):
        new_query = torch.cat(query.split(self.out_channels, dim=-1), dim=0) # Reshape the query
        new_key = torch.cat(key.split(self.out_channels, dim=-1), dim=0) # Reshape the key
        new_value = torch.cat(value.split(self.out_channels, dim=-1), dim=0) # Reshape the value
        sub_query, sub_key, sub_value = new_query.chunk(self.num_heads, dim=0), new_key.chunk(self.num_heads, dim=0), new_value.chunk(self.num_heads, dim=0) # Split the reshape tensor into multiple heads
 
        return sub_query, sub_key, sub_value
 
    def _scaled_dot_product_attention(self, query, key, value):
        sub_query = query.chunk(self.num_heads, dim=0) # Split the query tensor into multiple heads
        sub_key = key.chunk(self.num_heads, dim=0) # Split the key tensor into multiple heads
        sub_value = value.chunk(self.num_heads, dim=0) # Split the value tensor into multiple heads
        scaled_qk = torch.matmul(sub_query, sub_key.transpose(-2, -1)) # Compute the scaled dot product
        scaled_qk = scaled_qk.div(1. / self.in_channels ** 0.5) # Scale the scaled dot product
        softmax_qk = nn.functional.softmax(scaled_qk, dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        dropout_qk = dropout_qk.chunk(self.num_heads, dim=0) # Split the dropout output into multiple heads
        attention = torch.cat([d.unsqueeze(0) for d in dropout_qk], dim=0)  # Concatenate the output of multiple heads
        attention = attention.permute(1, 0, 2, 3).contiguous().view(attention.shape[1], -1, self.out_channels) # Rearrange the output for the linear transformation
        output = torch.matmul(attention, sub_value) # Compute the output of the linear transformation
        return output
 
    def _final_linear_projection(self, output):
        output = output.permute(1, 0, 2).contiguous() # Rearrange the output for the linear transformation
        output = output.view(-1, self.out_channels) # Reshape the output for the linear transformation
        output = self.fc(output) # Apply the linear transformation to the output
        return output
    
# Initializing the model
m = Model(hidden_size, hidden_size, 4, 0.01)

# Inputs to the model
query = torch.randn(2, 20, hidden_size)
key = torch.randn(2, 40, hidden_size)
value = torch.randn(2, 40, hidden_size)
