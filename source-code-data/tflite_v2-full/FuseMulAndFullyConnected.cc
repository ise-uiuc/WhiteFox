// Returns true if the value's element type is F32.
bool IsF32Value(Value value) {
  return value.getType().cast<ShapedType>().getElementType().isF32();
}

// Replace ..
// FC(Mul(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, Mul(filter, rhs), bias)
// .. if rhs, filter, and bias are all constants.
// The generated Mul will be constant folded to a single matrix.
struct FuseMulAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
    
    
    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Mul.
    auto mul_op =
        dyn_cast_or_null<TFL::MulOp>(fc_op.getInput().getDefiningOp());
    if (!mul_op) return failure();
    if (mul_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match muls where the multiplier constant is not 1D.
    {
      auto multiplier_shape = mul_op.getRhs().getType().cast<ShapedType>();
      if (!multiplier_shape.hasStaticShape()) return failure();
      if (multiplier_shape.getShape().size() != 1) return failure();
    }

    // We rely on constant folding, implemented only for F32. Check types.
    if (!IsF32Value(mul_op.getRhs()) || !IsF32Value(fc_op.getFilter())) {
      return failure();
    }

    auto location =
        FusedLoc::get(mul_op.getContext(), {mul_op.getLoc(), fc_op.getLoc()});

    auto new_filter = rewriter.create<TFL::MulOp>(
        location,
        /*lhs=*/fc_op.getFilter(),
        /*rhs=*/mul_op.getRhs(),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"));
    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        location, fc_op.getOutput().getTypes(),
        /*input=*/mul_op.getLhs(),
        /*filter=*/new_filter,
        /*bias=*/fc_op.getBias(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.getFusedActivationFunction()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.getKeepNumDims()),
        /*asymmetric_quantize_inputs=*/fc_op.getAsymmetricQuantizeInputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.getOutput());

    return success();
  }
};