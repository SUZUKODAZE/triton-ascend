//===- SimtIndirectAtomicCostModel.cpp - Loaded-index atomic service -----===//
//
// One atomic workload pays three things: a per-operation startup, one service
// interval per logical element, and the measured result-dependency latency when
// the returned value is used.  The rate, startup and dependency come from the
// profile slot that matches the operation, its element type and the address
// class; the class is `loaded_index` for a scatter whose offsets come from a
// loaded index tensor.
//
// GM bitwise atomics are not priced here.  They have no native instruction, and
// the reachable SoftAccel lowering reads each distinct address and commits with
// a compare-and-swap retry loop, so the service scales with the number of
// distinct updated addresses, which TTIR does not expose.  Those workloads are
// left unmodeled and contribute only the two extra vector-function starts that
// the template launches beyond the one the Stage already paid.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/Models/SimtIndirectAtomicCostModel.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::ascend {

double simtIndirectAtomicServiceCycles(
    llvm::ArrayRef<AtomicWorkload> workloads, const StageModeProfile &profile,
    double stageSetupCycles) {
  double service = 0.0;
  for (const AtomicWorkload &atomic : workloads) {
    if (atomic.kind == "and" || atomic.kind == "or" || atomic.kind == "xor") {
      service += 2.0 * stageSetupCycles;
      continue;
    }
    const llvm::StringRef addressClass =
        atomic.addressDependsOnLoadedIndex
            ? "loaded_index"
            : (atomic.provenContiguousRunWidth > 1.0 ? "contiguous"
                                                     : "unknown_address");
    auto rate = profile.atomicRates.find(atomic.profileKey() + "." +
                                         addressClass.str());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find(addressClass);
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find(atomic.profileKey());
    if (rate == profile.atomicRates.end())
      rate = profile.atomicRates.find("default");
    if (rate == profile.atomicRates.end())
      continue;
    const StageAtomicRate &atomicRate = rate->second;
    double base = atomic.logicalOperationInstances *
                      atomicRate.operationStartupCycles +
                  atomic.logicalElements / atomicRate.logicalElementsPerCycle;
    if (atomic.resultUsed)
      base += atomic.logicalOperationInstances *
              atomicRate.resultDependencyCycles;
    if (atomic.contentionUnknown)
      base *= atomicRate.unknownContentionMultiplier;
    service += base;
  }
  return service;
}

} // namespace mlir::ascend
