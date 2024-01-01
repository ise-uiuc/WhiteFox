namespace {
Status CanonicalizeNonTupleConditional(HloInstruction* conditional) {
  TF_RET_CHECK(conditional->opcode() == HloOpcode::kConditional);
  for (auto* branch : conditional->called_computations()) {
    HloInstruction* root = branch->root_instruction();
    TF_RET_CHECK(!root->shape().IsTuple());

    HloInstruction* tuple =
        branch->AddInstruction(HloInstruction::CreateTuple({root}));
    branch->set_root_instruction(tuple, /*accept_different_shape=*/true);
  }
  auto parent = conditional->parent();
  const Shape& root_shape = conditional->shape();
  auto new_shape = ShapeUtil::MakeTupleShape(absl::MakeSpan(&root_shape, 1));
  auto new_conditional =
      parent->AddInstruction(conditional->CloneWithNewShape(new_shape));
  auto gte = parent->AddInstruction(
      HloInstruction::CreateGetTupleElement(root_shape, new_conditional, 0));
  TF_RETURN_IF_ERROR(parent->ReplaceInstruction(conditional, gte));
  return OkStatus();
}
}  // namespace

StatusOr<bool> ConditionalCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    for (auto* inst : comp->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kConditional &&
          !inst->shape().IsTuple()) {
        TF_RETURN_IF_ERROR(CanonicalizeNonTupleConditional(inst));
        changed = true;
      }
    }
  }
  XLA_VLOG_LINES(
      2, "ConditionalCanonicalizer::Run(), after:\n" + module->ToString());
  return changed;
}
