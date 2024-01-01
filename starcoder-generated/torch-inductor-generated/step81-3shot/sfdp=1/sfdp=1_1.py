
class Model(torch.nn.Module):
    def __init__(self, input_shape, num_heads, sequence_length=128, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_p=0.0, attn_drop_p=0.0, drop_path_p=0.0):
        super().__init__()

        self.pos = torch.nn.Parameter(torch.zeros(1, sequence_length + 1, 8))

        self.conv_mask = torch.nn.Conv2d(21, 8, 1)

    def mlp(self, x):
        return torch.nn.functional.gelu(torch.nn.Linear(x, 2048))

    def forward(self, x):
        mask_logits = self.conv_mask(x)
        v = torch.sigmoid(mask_logits)

        return v

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
