// Fuses Unpack with proceeding Concatenation to Reshape if output type has
// static shape and activation function is none. For example:
//
//   // %input: tensor<1x3x2xf32>
//   %unpack:3 = "tfl.unpack"(%input) {axis = 1 : i32, num = 3 : i32}
//   %res = "tfl.concatenation"(%unpack#0, %unpack#1, %unpack#2)
//        {axis = -1 : i32, fused_activation_function = "NONE"}
//
// can be optimized to
//
//   %cst = arith.constant dense<[1, 6]> : tensor<2xi32>
//   %res = "tfl.reshape"(%input, %cst)
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

    auto new_shape_array = output_type.getShape();
    // This is to workaround the unnecessary cast i64 -> i32.
    SmallVector<int32_t, 4> new_shape_array_i32;
    for (auto size : new_shape_array) {
      new_shape_array_i32.push_back(
          ShapedType::isDynamic(size) ? -1 : static_cast<int32_t>(size));
    }
    auto new_shape = rewriter.create<TFL::ConstOp>(
        concat_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get(new_shape_array_i32.size(),
                                  rewriter.getIntegerType(32)),
            new_shape_array_i32));

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        concat_op, output_type, unpack_op.getInput(), new_shape);
    
    return success();
  }
};