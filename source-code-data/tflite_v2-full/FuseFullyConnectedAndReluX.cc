// TODO(b/136285429): Move to tablegen when variadic is supported.
template <typename ReluXOp, char const *Act>
struct FuseFullyConnectedAndReluX : public OpRewritePattern<ReluXOp> {
  using OpRewritePattern<ReluXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluXOp relu_op,
                                PatternRewriter &rewriter) const override {
    
    Operation *input = relu_op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return failure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.getFusedActivationFunction() != "NONE")
      return failure();

    auto new_activation_func = rewriter.getStringAttr(Act);
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.getWeightsFormat());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.getKeepNumDims());
    auto fc = rewriter.create<FullyConnectedOp>(
        FusedLoc::get(relu_op.getContext(),
                      {fully_connected_op.getLoc(), relu_op.getLoc()}),
        relu_op.getType(), /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/new_activation_func,
        /*weights_format=*/new_weights_format,
        /*keep_num_dims=*/new_keep_num_dims,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(relu_op, fc.getOutput());

    return success();
  }
};