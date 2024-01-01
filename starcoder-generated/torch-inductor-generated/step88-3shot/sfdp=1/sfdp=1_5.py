
class MultiLayerTransformerModel(torch.nn.Module):
    def __init__(self, d_model, num_heads, layer_count, dropout_p, input_dim, output_dim):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_count = layer_count
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.transformer_layers = torch.nn.ModuleList(
            [TransformerLayer(self.d_model, self.num_heads, dropout_p) for _ in range(self.layer_count)]))
        self.final_layer = torch.nn.Linear(self.d_model, self.output_dim)
    
    def forward(self, x1):
        output = x1.reshape(-1, self.input_dim)
        # The input dimension is 64 x 128 x 128, with batch size 64, but we need to flatten the last two tensor dimensions (128, 128)
        output = output.reshape(-1, self.input_dim)
        for k in range(len(self.transformer_layers)):
            output = self.transformer_layers[k](output)
        output = self.final_layer(output)
        output = output.reshape(shape[0], shape[-2], -1, 1, self.output_dim)
        # Re-shaping the dimension for the case where we use batch normalization afterwards
        return output.reshape(1, -1, 1, self.output_dim)

# Inputs to the model
x1 = torch.randn(1, 128, 64, 80)
