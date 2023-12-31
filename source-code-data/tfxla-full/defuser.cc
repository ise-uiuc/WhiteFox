namespace xla {

namespace {

// Copy all the instructions in the given fusion instruction into the fusion
// instruction's parent computation and replace the use of the fusion
// instruction with the copy of the fusion expression root.
Status Defuse(HloInstruction* fusion_instruction) {
  VLOG(2) << "Defusing instruction: " << fusion_instruction->ToString();

  HloComputation* fused_computation =
      fusion_instruction->fused_instructions_computation();

  // A map from fused instruction to its defused clone.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      defused_instructions;
  // Initialize map to contain the fusion instruction parameters mapping
  // to the operands of the fusion instruction.
  for (int64_t i = 0; i < fusion_instruction->operand_count(); ++i) {
    defused_instructions[fused_computation->parameter_instruction(i)] =
        fusion_instruction->mutable_operand(i);
  }

  // Create a clone of each instruction of the fused computation in the same
  // computation as the fusion instruction itself.
  // TODO(b/68227302): Moving instruction to new computation rather than
  // cloning and deleting.
  for (HloInstruction* fused_instruction :
       fused_computation->MakeInstructionPostOrder()) {
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    for (HloInstruction* operand : fused_instruction->operands()) {
      new_operands.push_back(defused_instructions.at(operand));
    }
    HloInstruction* defused_instruction =
        fusion_instruction->parent()->AddInstruction(
            fused_instruction->CloneWithNewOperands(fused_instruction->shape(),
                                                    new_operands));
    defused_instructions[fused_instruction] = defused_instruction;
  }

  TF_RETURN_IF_ERROR(fusion_instruction->ReplaceAllUsesWith(
      defused_instructions.at(fusion_instruction->fused_expression_root())));

  HloModule* module = fusion_instruction->GetModule();
  TF_RETURN_IF_ERROR(
      fusion_instruction->parent()->RemoveInstruction(fusion_instruction));
  return module->RemoveEmbeddedComputation(fused_computation);
}

}  // namespace

StatusOr<bool> Defuser::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(1) << "Defusing module " << module->name();
  XLA_VLOG_LINES(2, "Before defusion:\n" + module->ToString());

  bool changed = false;
  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);
  TF_RETURN_IF_ERROR(call_graph->VisitNodes(
      [&](const CallGraphNode& call_graph_node) -> Status {
        if (call_graph_node.computation()->IsFusionComputation()) {
          TF_RET_CHECK(call_graph_node.caller_callsites().size() == 1);
          HloInstruction* fusion_instruction =
              call_graph_node.caller_callsites()[0].instruction();
          TF_RETURN_IF_ERROR(Defuse(fusion_instruction));
          changed = true;
        }
        return OkStatus();
      },
      /*visit_unreachable_nodes=*/true));

  XLA_VLOG_LINES(2, "After defusion:\n" + module->ToString());

  return changed;
}

}  // namespace xla
