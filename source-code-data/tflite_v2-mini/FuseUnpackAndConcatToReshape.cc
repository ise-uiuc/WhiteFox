// Fuses Unpack with proceeding Concatenation to Reshape if output type has
// static shape and activation function is none.
struct FuseUnpackAndConcatToReshape
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat_op,
                                PatternRewriter &rewriter) const override {

    if (concat_op.getFusedActivationFunction() != "NONE") {
      return failure();
    }

    // Checks all operands come from the same unpack op.
    auto first_operand = concat_op.getValues().front();
    auto unpack_op =
        dyn_cast_or_null<TFL::UnpackOp>(first_operand.getDefiningOp());
    if (!unpack_op || unpack_op.getNumResults() != concat_op.getNumOperands()) {
      return failure();
    }
    for (const auto &index_and_value : llvm::enumerate(concat_op.getValues())) {
      if (index_and_value.value() !=
          unpack_op.getResult(index_and_value.index())) {
        return failure();
      }
    }

    auto output_type = concat_op.getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) {
      return failure();
    }
    
    return success();
  }
};