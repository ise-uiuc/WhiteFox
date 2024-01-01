StatusOr<bool>
BatchDotSimplification::ElideDegenerateBatchDimensionFromBatchDot(
    HloInstruction* batch_dot) {
  // This pass assumes the lhs and rhs batch dimensions are equal and strictly
  // ascending.
  const auto& is_iota = [](absl::Span<const int64_t> dims) {
    for (int64_t i = 0; i < dims.size(); ++i) {
      if (dims[i] != i) {
        return false;
      }
    }
    return true;
  };
  if (!absl::c_equal(
          batch_dot->dot_dimension_numbers().lhs_batch_dimensions(),
          batch_dot->dot_dimension_numbers().rhs_batch_dimensions()) ||
      !is_iota(batch_dot->dot_dimension_numbers().lhs_batch_dimensions())) {
    return false;
  }

  const DotDimensionNumbers& dim_numbers = batch_dot->dot_dimension_numbers();
  HloInstruction *lhs = batch_dot->mutable_operand(0),
                 *rhs = batch_dot->mutable_operand(1);
  const Shape& lhs_shape = lhs->shape();

  // A dot with no contracting dims will be rewritten into a multiply by
  // AlgebraicSimplifier. Dots with multiple contracting dims are currently
  // unsupported.
  if (dim_numbers.lhs_contracting_dimensions_size() != 1) {
    return false;
  }

  std::vector<int64_t> degenerate_dims;
  for (int64_t batch_dim : dim_numbers.lhs_batch_dimensions()) {
    if (lhs_shape.dimensions(batch_dim) == 1) {
      degenerate_dims.push_back(batch_dim);
    }
  }

  if (degenerate_dims.empty()) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      ElideDegenerateDims(lhs, degenerate_dims));
  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      ElideDegenerateDims(rhs, degenerate_dims));

  DotDimensionNumbers new_dim_numbers = dim_numbers;
  new_dim_numbers.clear_lhs_batch_dimensions();
  new_dim_numbers.clear_rhs_batch_dimensions();

  for (int64_t i = 0, e = dim_numbers.lhs_batch_dimensions_size() -
                          degenerate_dims.size();
       i < e; i++) {
    new_dim_numbers.add_lhs_batch_dimensions(i);
    new_dim_numbers.add_rhs_batch_dimensions(i);
  }

  new_dim_numbers.set_lhs_contracting_dimensions(
      0,
      new_dim_numbers.lhs_contracting_dimensions(0) - degenerate_dims.size());
  new_dim_numbers.set_rhs_contracting_dimensions(
      0,
      new_dim_numbers.rhs_contracting_dimensions(0) - degenerate_dims.size());

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_dot,
      MakeDotHlo(new_lhs, new_rhs, new_dim_numbers,
                 batch_dot->precision_config(),
                 /*preferred_element_type=*/batch_dot->shape().element_type()));

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot_reshaped,
                      MakeReshapeHlo(batch_dot->shape(), new_dot));

  VLOG(2) << "Replaced " << batch_dot->ToString() << " with "
          << new_dot->ToString();

  TF_RETURN_IF_ERROR(
      batch_dot->parent()->ReplaceInstruction(batch_dot, new_dot_reshaped));

  return true;
}
