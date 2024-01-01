// Remove Reshape before FullyConnected when `keep_num_dims=false` and Reshape
// does not alter the last dimension as FullyConnected will collapse all other
// dimensions into a single dimension. For example,
//
//   %shape = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
//   %reshape = tfl.reshape(%input, %shape) // %input: tensor<128x64xf32>
//   %fc = tfl.fully_connected(%reshape, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
struct RemoveReshapeBeforeFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fully_connected_op,
                                PatternRewriter &) const override {

    auto input = fully_connected_op.getInput();
    auto input_ty = input.getType().dyn_cast<ShapedType>();
    auto output_ty = fully_connected_op.getOutput()[0]
                         .getType()
                         .template dyn_cast<ShapedType>();
    if (!input_ty.hasStaticShape() ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims() || !output_ty.hasStaticShape() ||
        output_ty.getRank() != 2) {
      return failure();
    }

    auto reshape_op = input.getDefiningOp<TFL::ReshapeOp>();
    if (!reshape_op) return failure();

    // Check if the last dimension does not change after reshape.
    auto reshape_input = reshape_op.getInput();
    auto reshape_input_ty = reshape_input.getType().dyn_cast<ShapedType>();
    if (!reshape_input_ty.hasStaticShape() || input_ty.getRank() == 0 ||
        reshape_input_ty.getRank() == 0 ||
        input_ty.getDimSize(input_ty.getRank() - 1) !=
            reshape_input_ty.getDimSize(reshape_input_ty.getRank() - 1)) {
      return failure();
    }

    return success();
  }
};