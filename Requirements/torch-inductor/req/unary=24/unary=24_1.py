t1 = conv(input_tensor) # Apply pointwise convolution with kernel size 1 to the input tensor
t2 = t1 > 0 # Create a boolean mask where each element is True if the corresponding element in t1 is greater than 0, False otherwise
t3 = t1 * negative_slope # Multiply the output of the convolution by the negative_slope
t4 = torch.where(t2, t1, t3) # Apply the where function to select elements from t1 or t3 based on the mask t2
