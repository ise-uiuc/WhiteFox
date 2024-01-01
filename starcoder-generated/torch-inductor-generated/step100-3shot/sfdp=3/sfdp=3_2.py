
class Model(torch.nn.Module):
    def __init__(self, num_attn_heads, num_input_features, dropout_p):
        super().__init__()
        d_model = num_input_features * num_attn_heads
        self.query = torch.nn.Linear(num_input_features, d_model)
        self.key = torch.nn.Linear(num_input_features, d_model)
        self.value = torch.nn.Linear(num_input_features, d_model)
        self.scale_factor = torch.nn.Parameter(
            torch.sqrt(torch.Tensor([1.0 / num_attn_heads])),
            requires_grad=True
        )
        self.num_attn_heads = num_attn_heads
        self.dropout_p = dropout_p
 
    def forward(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
 
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(self.scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(
            softmax_qk, p=self.dropout_p,
            training=self.training
        )
        output = dropout_qk.matmul(v)
 
        output_size = inputs.size()[:-1] + (self.num_attn_heads, num_input_features)
        output_per_head = output.view(output_size)
        output = output_per_head.permute(0, 2, 1, 3).contiguous()
        output = output.view(output_size[0], -1, num_input_features)
        return output

# Initializing the model
m = Model(1, 3, 0.2)

# Inputs to the model
inputs = torch.randn(2, 3, 3)
