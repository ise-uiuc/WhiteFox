// Concat(Concat(A, B), C) => Concat(A, B, C)
StatusOr<bool> ConcatForwarding(HloInstruction* concat) {
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }
  bool changed = false;

  auto parent = concat->parent();
  std::vector<HloInstruction*> new_operands;
  for (HloInstruction* operand : concat->operands()) {
    if (operand->opcode() != HloOpcode::kConcatenate ||
        operand->concatenate_dimension() != concat->concatenate_dimension()) {
      new_operands.push_back(operand);
    } else {
      changed = true;
      for (HloInstruction* operand_operand : operand->operands()) {
        new_operands.push_back(operand_operand);
      }
    }
  }
  if (changed) {
    auto new_concat = parent->AddInstruction(HloInstruction::CreateConcatenate(
        concat->shape(), new_operands, concat->concatenate_dimension()));
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(concat, new_concat));
  }
  return changed;
}