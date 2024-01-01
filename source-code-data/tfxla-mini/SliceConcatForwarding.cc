// Slice(Concat(A1, A2, ..., An, ...), [n:n+1]) => An
StatusOr<bool> SliceConcatForwarding(HloInstruction* slice) {
  if (slice->opcode() != HloOpcode::kSlice) {
    return false;
  }
  auto concat = slice->mutable_operand(0);
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }

  if (slice->shape().rank() != 1) {
    // Slice concat forwarding only work for size 1 tensor.
    return false;
  }

  int64_t concat_dim = concat->concatenate_dimension();

  std::vector<HloInstruction*> new_operands;
  int64_t size_so_far = 0;
  int64_t slice_size = slice->shape().dimensions(concat_dim);
  if (slice_size != slice->slice_limits(0) - slice->slice_starts(0)) {
    return false;
  }
  if (slice->slice_strides(0) != 1) {
    return false;
  }
  for (HloInstruction* operand : concat->operands()) {
    if (size_so_far == slice->slice_starts(0) &&
        operand->shape().dimensions(0) == slice_size) {
      // Found an operand that can be forwarded.
      TF_RETURN_IF_ERROR(slice->ReplaceAllUsesWith(operand));
      return true;
    }
    size_so_far += operand->shape().dimensions(concat_dim);
  }
  return false;
}