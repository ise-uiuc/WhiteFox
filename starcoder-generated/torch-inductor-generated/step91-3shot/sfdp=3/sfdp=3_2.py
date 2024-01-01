

class Model(torch.nn.Module):
  ...
    def scaled_dot_product_attention(self, query, key, value, mask=None, scale_factor=1/sqrt(query.shape[-1])):
        