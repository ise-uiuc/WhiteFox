
def custom_relu_with_static_slope(input_tensor<fim_suffix>, mask):
    return torch.where(
        input_tensor > 0,
        input_tensor,
        input_tensor * negative_slope,
    )
