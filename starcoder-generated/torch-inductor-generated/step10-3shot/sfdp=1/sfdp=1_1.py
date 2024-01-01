
class Model(torch.nn.Module):
    def __init__(self, num_attention_heads, input_shape):
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_head_projections = 3
        self.input_shape = input_shape
        self.query = torch.nn.Parameter(torch.randn(self.num_heads, input_shape[0], self.num_head_projections), requires_grad=True)
        self.key = torch.nn.Parameter(torch.randn(self.num_heads, input_shape[0], self.num_head_projections), requires_grad=True)
        self.value = torch.nn.Parameter(torch.randn(self.num_heads, input_shape[0], self.num_head_projections), requires_grad=True)

    def forward(self, x1, dropout_p):
        inv_scale_factor = torch.sqrt(torch.tensor(input_shape[0]).float())
        q = torch.matmul(self.query, x1.transpose(-1, -2)).view(self.num_heads, x1.shape[1], self.num_head_projections);
        k = torch.matmul(self.key, x1.transpose(-1, -2)).view(self.num_heads, x1.shape[1], self.num_head_projections);
        v = torch.matmul(self.value, x1.transpose(-1, -2)).view(self.num_heads, x1.shape[1], self.num_head_projections);

        qk = torch.matmul(q, k.transpose(0, 1))
        scaled_qk = qk.div(inv_scale_factor)
        softmax_qk = scaled_qk.softmax(-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = torch.matmul(dropout_qk, v);
        output = output.view(output.shape[1], output.shape[2], output.shape[3]).transpose(-1, -2)
    
        return output

def attention_mask_func(query_shape, key_length=None, query_length=None):
    if query_length is None:
        query_length = query_shape[1]
    if key_length is None:
        key_length = query_length
    attention_mask_elements = torch.zeros(query_shape[0], key_length)
    for batch in range(query_shape[0]):
        attention_mask_elements[batch, :query_shape[1]] = 1
    return attention_mask_elements

# Initializing the model
m = Model(4, [128, 512])

# Inputs to the model
x1 = torch.randn(8, 512, 128)
dropout_p = 0.1
hidden_states = m(x1, dropout_p)
