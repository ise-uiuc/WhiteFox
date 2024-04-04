PreservedAnalyses DeadArgumentEliminationPass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  bool Changed = false;

  // Finally, look for any unused parameters in functions with non-local
  // linkage and replace the passed in parameters with poison.
  for (auto &F : M)
    Changed |= removeDeadArgumentsFromCallers(F);

  if (!Changed)
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

bool DeadArgumentEliminationPass::removeDeadArgumentsFromCallers(Function &F) {

  // Functions with local linkage should already have been handled, except if
  // they are fully alive (e.g., called indirectly) and except for the fragile
  // (variadic) ones. In these cases, we may still be able to improve their
  // statically known call sites.
  if ((F.hasLocalLinkage() && !LiveFunctions.count(&F)) &&
      !F.getFunctionType()->isVarArg())
    return false;

  // Don't touch naked functions. The assembly might be using an argument, or
  // otherwise rely on the frame layout in a way that this analysis will not
  // see.
  if (F.hasFnAttribute(Attribute::Naked))
    return false;

  if (F.use_empty())
    return false;

  AttributeMask UBImplyingAttributes =
      AttributeFuncs::getUBImplyingAttributes();
  for (Argument &Arg : F.args()) {
    if (!Arg.hasSwiftErrorAttr() && Arg.use_empty() &&
        !Arg.hasPassPointeeByValueCopyAttr()) {
      if (Arg.isUsedByMetadata()) {
        Arg.replaceAllUsesWith(PoisonValue::get(Arg.getType()));
        LLVM_DEBUG(dbgs() << "Arg.replaceAllUsesWith(PoisonValue::get(Arg.getType()));\n");
        Changed = true;
      }
      UnusedArgs.push_back(Arg.getArgNo());
      F.removeParamAttrs(Arg.getArgNo(), UBImplyingAttributes);
    }
  }

  if (UnusedArgs.empty())
    return false;

  return Changed;
}

