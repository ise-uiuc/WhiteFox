
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.mha_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.mha_attn_dropout = 0.1
        bias = True
        self.mha = torch.nn.MultiheadAttention(hidden_size, num_heads, bias=bias)
 
    def forward(self, input_tensor, attention_mask):
        attention_mask = torch.cat((input_tensor.new_zeros(input_tensor.size(0), 1, input_tensor.size(1)), input_tensor.new_ones(input_tensor.size(0), attention_mask.size(1) - 1, input_tensor.size(1))), 1)
        attention_mask = (1.0 - attention_mask) * -10000000.0
        attention_mask = attention_mask.unsqueeze(1)
 
        output = self.mha_layer_norm(input_tensor)
 
        output, output_weights = self.mha(output, output, output, attention_mask=attention_mask)
        output = output.transpose(0, 1).contiguous().view(input_tensor.size(1), -1)
 
        return output

# Initializing the model
m = Model(hidden_size=100, num_heads=5)

# Inputs to the model
input_tensor = torch.randn(4, 10, 100)
attention_mask = torch.rand(4, 10)
