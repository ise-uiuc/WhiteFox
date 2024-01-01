// Replace ..
// FC(Add(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, filter, FC(rhs, filter, bias))
// .. if rhs, filter, and bias are all constants.
// The second FC will be constant folded to a single vector.
struct FuseAddAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {

    // This only works with default format.
    if (fc_op.getWeightsFormat() != "DEFAULT") return failure();

    // Match Add.
    auto add_op =
        dyn_cast_or_null<TFL::AddOp>(fc_op.getInput().getDefiningOp());
    if (!add_op) return failure();
    if (add_op.getFusedActivationFunction() != "NONE") return failure();

    // Don't match adds where the added constant is not 1D.
    {
      auto addend_shape = add_op.getRhs().getType().cast<ShapedType>();
      if (!addend_shape.hasStaticShape()) return failure();
      if (addend_shape.getShape().size() != 1) return failure();
    }

    // Calculate new bias.  Generate a new FC; it will be constant folded.
    auto old_bias = fc_op.getBias();
    if (!old_bias || old_bias.getType().isa<NoneType>()) {
      // TODO(b/180752069): Figure out new bias' type when old bias is empty.
      return failure();
    }

    // The FC relies on constant folding, which is implemented on F32. Checks
    // types to be F32.
    {
      if (!IsF32Value(add_op.getRhs()) || !IsF32Value(fc_op.getFilter()) ||
          !IsF32Value(old_bias))
        return failure();
    }

    return success();
  }
};