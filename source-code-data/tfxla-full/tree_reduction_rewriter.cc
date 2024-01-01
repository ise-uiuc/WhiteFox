namespace xla {

class ReductionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  explicit ReductionRewriterVisitor(int64_t reduce_window_size)
      : reduce_window_size_(reduce_window_size) {}

  Status HandleReduce(HloInstruction *hlo) override {
    HloInstruction *reduced_op = hlo->mutable_operand(0);
    HloInstruction *initial_value = hlo->mutable_operand(1);
    const Shape &input_shape = reduced_op->shape();
    const Shape &reduce_shape = hlo->shape();

    if (!reduce_shape.IsArray()) {
      // TODO(b/210786051): Implement tree reduction rewrite for variadic
      // reductions on CPU as well.
      VLOG(1) << "Skipping rewrite for variadic reduction";
      return OkStatus();
    }

    // All of the reduced dimensions is smaller than the window size,
    // do not perform the rewrite.
    if (absl::c_all_of(hlo->dimensions(), [&](int64_t reduced_dim) {
          return input_shape.dimensions(reduced_dim) <= reduce_window_size_;
        })) {
      VLOG(1) << "Skipping tree reduction rewrite: all reduced dimensions are "
                 "smaller than "
              << reduce_window_size_;
      return OkStatus();
    }

    std::vector<int64_t> window_dimensions;
    std::vector<int64_t> window_strides;
    for (int64_t dim_idx = 0; dim_idx < input_shape.rank(); dim_idx++) {
      if (!absl::c_linear_search(hlo->dimensions(), dim_idx)) {
        window_dimensions.push_back(1);
        window_strides.push_back(1);
        continue;
      }

      int64_t window_size_for_dim =
          std::min(input_shape.dimensions(dim_idx), reduce_window_size_);

      window_dimensions.push_back(window_size_for_dim);
      window_strides.push_back(window_size_for_dim);
    }

    std::vector<std::pair<int64_t, int64_t>> padding =
        MakePadding(input_shape.dimensions(), window_dimensions, window_strides,
                    Padding::kSame);

    TF_ASSIGN_OR_RETURN(
        Window window, ShapeInference::InferWindowFromDimensions(
                           window_dimensions, window_strides, padding, {}, {}));

    TF_ASSIGN_OR_RETURN(Shape intermediate_shape,
                        ShapeInference::InferReduceWindowShape(
                            input_shape, initial_value->shape(), window));

    HloInstruction *reduce_window =
        hlo->parent()->AddInstruction(HloInstruction::CreateReduceWindow(
            intermediate_shape, reduced_op, initial_value, window,
            hlo->to_apply()));

    std::unique_ptr<HloInstruction> new_output =
        HloInstruction::CreateReduce(reduce_shape, reduce_window, initial_value,
                                     hlo->dimensions(), hlo->to_apply());

    return ReplaceWithNewInstruction(hlo, std::move(new_output));
  }

 private:
  int64_t reduce_window_size_;
};

StatusOr<bool> TreeReductionRewriter::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  ReductionRewriterVisitor visitor(reduce_window_size_);
  bool changed = false;
  for (const auto &computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    changed |= visitor.changed();
  }

  return changed;
}

}  // end namespace xla
