
def foo(input_x):
    relu1_out = F.relu(
        torch.tanh(
            F.conv2d( # Apply pointwise convolution with kernel size 1 to the input tensor
                F.relu(
                    F.dropout(
                        F.conv2d(F.relu(F.conv2d(input_x, torch.zeros([16, 16, 1, 1]))),torch.zeros([16, 16, 1, 1])),p=0.5)
                    )
              , torch.zeros([16, 16, 1, 1])
            )
        )
    )

    relu6_out = F.relu(
        F.max_pool2d(
            torch.tanh(
                F.conv2d(
                    relu1_out,torch.zeros([16, 16, 3, 3])
                )
            ), stride=2, kernel_size=3)
    )

    relu8_out = torch.tanh(
        F.interpolate(
            F.conv2d(relu1_out, torch.zeros([16, 16, 1, 1])),
            scale_factor=2
        )
    )

    relu10_out = relu6_out + relu8_out

    relu11_out = F.relu6(
        F.conv2d(
            relu10_out, torch.zeros([16, 16, 3, 3])
        )
    )

    return relu11_out
# Inputs to the model
x = torch.randn(1, 16, 100, 100)
