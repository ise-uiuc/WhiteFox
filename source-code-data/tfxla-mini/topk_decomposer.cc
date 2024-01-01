class TopkDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleCustomCall(HloInstruction* inst) override {
    HloCustomCallInstruction* call = DynCast<HloCustomCallInstruction>(inst);
    HloComputation* comp = inst->parent();
    if (call == nullptr || call->custom_call_target() != "TopK") {
      return OkStatus();
    }

    HloInstruction* input = call->mutable_operand(0);
    Shape iota_shape = input->shape();
    iota_shape.set_element_type(S32);
    size_t sort_dimension = input->shape().dimensions_size() - 1;
    std::vector<int64_t> zeroes(iota_shape.rank(), 0);
    std::vector<int64_t> ones(iota_shape.rank(), 1);
    HloComputation* comparator = call->to_apply();
    // Apply a slice to a tuple.
    auto slice_tuple = [&](HloInstruction* sort, const size_t index) {
      return comp->AddInstruction(HloInstruction::CreateSlice(
          call->shape().tuple_shapes(index),
          comp->AddInstruction(HloInstruction::CreateGetTupleElement(
              sort->shape().tuple_shapes(index), sort, index)),
          zeroes, call->shape().tuple_shapes(index).dimensions(), ones));
    };
    CHECK_NE(comparator, nullptr);
    // If only the topk values are necessary, skip the iota.
    if (call->user_count() == 1 && call->users().front()->tuple_index() == 0) {
      HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
          {input->shape()}, sort_dimension, {input}, call->to_apply(),
          /*is_stable=*/true));
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call->users().front(),
          comp->AddInstruction(HloInstruction::CreateSlice(
              call->shape().tuple_shapes(0), sort, zeroes,
              call->shape().tuple_shapes(0).dimensions(), ones))));
      sort->set_metadata(call->metadata());
    } else {
      HloInstruction* iota = comp->AddInstruction(
          HloInstruction::CreateIota(iota_shape, iota_shape.rank() - 1));
      HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({input->shape(), iota_shape}),
          sort_dimension, {input, iota}, call->to_apply(),
          /*is_stable=*/true));
      TF_RETURN_IF_ERROR(ReplaceInstruction(
          call, comp->AddInstruction(HloInstruction::CreateTuple(
                    {slice_tuple(sort, 0), slice_tuple(sort, 1)}))));
      sort->set_metadata(call->metadata());
    }
    return OkStatus();
  }
};


