// Remove Reshape after FullyConnected when `keep_num_dims=false`, the Reshape
// does not alter the last dimension and it restores the batch dimensions
// collapsed by the FullyConnected op due to `keep_num_dims=false`. For example,
//
//   // %input: tensor<4x16x32xf32>
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//   %shape = arith.constant dense<[4, 16, 32]> : tensor<3xi32>
//   %rs = tfl.reshape(%fc, %shape)
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = true, weights_format = "DEFAULT"}
struct RemoveReshapeAfterFullyConnected
    : public OpRewritePattern<TFL::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape_op,
                                PatternRewriter &rewriter) const override {

    auto fully_connected_op = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape_op.getInput().getDefiningOp());
    if (!fully_connected_op || fully_connected_op.getNumResults() != 1 ||
        fully_connected_op.getWeightsFormat() != "DEFAULT" ||
        fully_connected_op.getKeepNumDims())
      return failure();
    if (!reshape_op.getInput().hasOneUse()) return failure();

    auto input_shape =
        fully_connected_op.getInput().getType().cast<ShapedType>();
    auto output_shape = fully_connected_op.getType(0).cast<ShapedType>();
    auto reshape_shape = reshape_op.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape() || !output_shape.hasStaticShape() ||
        !reshape_shape.hasStaticShape())
      return failure();

    // Check that the reshape doesn't modify the last dimension and it restores
    // the input (batch) dimension with the exception of the feature (last)
    // dimension.
    if (output_shape.getShape().empty() || reshape_shape.getShape().empty() ||
        output_shape.getShape().back() != reshape_shape.getShape().back() ||
        input_shape.getShape().drop_back() !=
            reshape_shape.getShape().drop_back())
      return failure();

    llvm::SmallVector<Type, 1> output_type{reshape_op.getType()};
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        reshape_op, output_type, /*input=*/fully_connected_op.getInput(),
        /*filter=*/fully_connected_op.getFilter(),
        /*bias=*/fully_connected_op.getBias(),
        /*fused_activation_function=*/
        fully_connected_op.getFusedActivationFunction(),
        /*weights_format=*/fully_connected_op.getWeightsFormat(),
        /*keep_num_dims=*/true,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.getAsymmetricQuantizeInputsAttr());
    
    return success();
  }
};