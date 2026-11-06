//===- SimtIndirectAtomicCostModel.h - Loaded-index atomic service -------===//
//
// Service cost of a loaded-index atomic Stage.  The shape lives in
// SimtIndirectAtomicCostModel.cpp; every number comes from the profile's atomic
// rate slots.  Nothing here reads an operator or kernel name: an atomic
// workload is priced from its static description alone.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ASCEND_MODEL_ROUTEMODEL_SIMTINDIRECTATOMICCOSTMODEL_H
#define MLIR_ASCEND_MODEL_ROUTEMODEL_SIMTINDIRECTATOMICCOSTMODEL_H

#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::ascend {

/// Cycles charged for the atomic service of one Stage: the sum over its atomic
/// workloads of the per-operation startup, one service interval per logical
/// element, and the measured result-dependency latency when the returned value
/// is used, scaled by the unknown-contention multiplier.
///
/// GM bitwise atomics have no native instruction.  Their reachable drop is the
/// SoftAccel template, whose service depends on the number of distinct updated
/// addresses and is deliberately left unmodeled; those workloads contribute
/// only the two extra vector-function starts the template launches beyond the
/// one the Stage already paid.
double simtIndirectAtomicServiceCycles(
    llvm::ArrayRef<AtomicWorkload> workloads, const StageModeProfile &profile,
    double stageSetupCycles);

} // namespace mlir::ascend

#endif // MLIR_ASCEND_MODEL_ROUTEMODEL_SIMTINDIRECTATOMICCOSTMODEL_H
