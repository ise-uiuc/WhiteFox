
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, z, inv_scale_factor, dropout_p):
        return torch.nn.functional.multi_head_attention_forward(
            query=x,
            key=y,
            value=z,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.2,
            out_proj_weight=None,
            out_proj_bias=None,
            use_separate_proj_weight=False,
            training=False,
            dropout_state=None,
            find_unused_parameters=True)

# Initializing the model
m = Model()

# Input tensors to the model
x = torch.randn(2, 3, 4, 20)
y = torch.randn(2, 5, 6, 16)
z = torch.randn(2, 5, 6, 16)
