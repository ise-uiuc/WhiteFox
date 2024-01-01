
class Model(
    torch.nn.Module
):  # Note this implementation modifies the key tensor to avoid an index out of bounds error, and also is different in that we are concatenating instead of multiplying the tensors
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(
            torch.randn(
                4, 128, 128
            )  # 128 features in the key tensor and 4 layers in the encoder, leading to a key tensor of size [4 x 128 x 128]
        )

    def forward(self, x1):
        q = x1
        k = self.key.view(4, 128, 128)
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(64, 128, 24, 24)
