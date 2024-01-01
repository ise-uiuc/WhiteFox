namespace xla {
namespace {

// Simplifies floating-point conversions `A -> B -> C -> D` as `A -> D`.
StatusOr<bool> RunOnComputation(HloComputation& computation) {
  bool changed = false;
  for (HloInstruction* instruction : computation.MakeInstructionPostOrder()) {
    HloInstruction* input = instruction;
    size_t convert_chain_length = 0;

    while ((input->opcode() == HloOpcode::kConvert) &&
           primitive_util::IsFloatingPointType(input->shape().element_type())) {
      input = input->mutable_operand(0);
      ++convert_chain_length;
    }

    if (convert_chain_length < 2) continue;

    if (instruction->shape().element_type() == input->shape().element_type()) {
      TF_RETURN_IF_ERROR(
          instruction->parent()->ReplaceInstruction(instruction, input));
    } else {
      TF_RETURN_IF_ERROR(instruction->parent()->ReplaceWithNewInstruction(
          instruction,
          HloInstruction::CreateConvert(instruction->shape(), input)));
    }
    changed = true;
  }
  return changed;
}

}  // namespace

StatusOr<bool> SimplifyFPConversions::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(*computation));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
