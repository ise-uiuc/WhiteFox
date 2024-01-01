
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_p, output_dropout_p):
        super().__init__()
        self.attention_dropout = torch.nn.Dropout(attention_dropout_p)
        self.attention_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.attention_output_bias = torch.nn.Parameter(torch.zeros(num_attention_heads, hidden_size))
        self.attention_output_dropout = torch.nn.Dropout(output_dropout_p)
        self.project = torch.nn.Linear(hidden_size, hidden_size)
        self.output_layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, attention_mask):
        attention_output = F.multi_head_attention_forward(
            query,
            key,
            value,
            torch.empty([0]),
            torch.cat([attention_mask, attention_mask], dim=1),
            self.attention_output_bias,
            3,
            0,
            "",
            True,
            torch.empty([0]),
            False,
            True,
            self.attention_dropout.p,
        )
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_layer_norm(value + attention_output)
        proj_value = self.project(attention_output)
        proj_value = self.output_layer_norm(attention_output + proj_value)
        return proj_value

# Initializing the model
m = Model(hidden_size=768, num_attention_heads=8, attention_dropout_p=0.1, output_dropout_p=0.1)

# Inputs to the model
x1 = torch.randn(1, 128, 768)
x2 = torch.randn(128, 128, 768)
x3 = torch.randn(128, 128, 768)
