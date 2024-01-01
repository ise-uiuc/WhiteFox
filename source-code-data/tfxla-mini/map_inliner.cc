Status MapInlinerVisitor::HandleMap(HloInstruction* map) {
  HloComputation* function = map->to_apply();
  HloInstruction& root = *function->root_instruction();
  // Only inlining functions that are simply a single operation until a better
  // profitability model for inlining is defined.
  if (hlo_query::AllOperandsAreParameters(root)) {
    if (root.opcode() == HloOpcode::kFusion) {
      // Cloning not supported for these instructions.
      return OkStatus();
    }
    VLOG(10) << "inlining map({X ... Y}, op) => : op(X ... Y) with function "
             << root.ToShortString();
    if (root.opcode() == HloOpcode::kParameter) {
      // If the root is a parameter, then use the corresponding operand as the
      // result of the computation.
      TF_RETURN_IF_ERROR(
          map->ReplaceAllUsesWith(map->operands()[root.parameter_number()]));
      TF_RETURN_IF_ERROR(computation_->RemoveInstruction(map));
    } else if (root.opcode() == HloOpcode::kConstant) {
      // If the input is a constant then the shape of the constant could be
      // different than the map shape. Hence, a broadcast is needed, else the
      // cloned operand with new shape and operands work.
      //
      // The constant is in an embedded computation and needs to be recreated
      // as part of the computation that the broadcast is inserted into.
      HloInstruction* constant = computation_->AddInstruction(root.Clone());
      HloInstruction* placed_instruction = computation_->AddInstruction(
          HloInstruction::CreateBroadcast(map->shape(), constant, {}));
      TF_RETURN_IF_ERROR(
          computation_->ReplaceInstruction(map, placed_instruction));
    } else {
      std::vector<HloInstruction*> params;
      for (int64_t o = 0; o < root.operands().size(); o++) {
        params.push_back(map->operands()[root.operand(o)->parameter_number()]);
      }
      HloInstruction* placed_instruction = computation_->AddInstruction(
          root.CloneWithNewOperands(map->shape(), params));
      TF_RETURN_IF_ERROR(
          computation_->ReplaceInstruction(map, placed_instruction));
    }
    changed_ = true;
    return OkStatus();
  }

  return OkStatus();
}
