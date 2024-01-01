namespace xla {

StatusOr<bool> ZeroSizedHloElimination::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instruction : comp->MakeInstructionPostOrder()) {
      if (instruction->HasSideEffect() || !instruction->shape().IsArray() ||
          instruction->opcode() == HloOpcode::kConstant) {
        continue;
      }
      if (comp->IsSafelyRemovable(instruction) &&
          ShapeUtil::IsZeroElementArray(instruction->shape()) &&
          instruction->shape().is_static()) {
        // If the instruction doesn't have a layout, use a default layout for
        // the literal.
        Shape shape = instruction->shape();
        if (!LayoutUtil::HasLayout(shape)) {
          LayoutUtil::SetToDefaultLayout(&shape);
        }
        TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
            instruction,
            HloInstruction::CreateConstant(Literal::CreateFromShape(shape))));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
