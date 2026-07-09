/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>

static constexpr const char *DEBUG_TYPE = "BufferCountManager";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)
#define DBG_PRINT(...) do { fprintf(stderr, "[BCM] " __VA_ARGS__); fflush(stderr); } while(0)

namespace mlir {
namespace triton {

namespace {

constexpr int kDefaultIntraBufferCount = 2;
constexpr int kDefaultInterBufferCount = 1;
constexpr int kDefaultLoadBufferCount = 1;

inline llvm::StringLiteral getAttrName(BufferCountManager::DepType type) {
    switch (type) {
        case BufferCountManager::DepType::IntraCore:
            return CVPipeline::kIntraBufCount;
        case BufferCountManager::DepType::InterCore:
            return CVPipeline::kInterCoreBufCount;
        case BufferCountManager::DepType::LoadStore:
            return CVPipeline::kLoadStoreBufCount;
    }
    llvm_unreachable("unknown BufferCountManager::DepType");
}

inline int getDefaultCount(BufferCountManager::DepType type) {
    switch (type) {
        case BufferCountManager::DepType::IntraCore:   return kDefaultIntraBufferCount;
        case BufferCountManager::DepType::InterCore:   return kDefaultInterBufferCount;
        case BufferCountManager::DepType::LoadStore:   return kDefaultLoadBufferCount;
    }
    llvm_unreachable("unknown BufferCountManager::DepType");
}

} // namespace

constexpr int kBufferCountWarningThreshold = 3;

BufferCountManager::BufferCountManager(Operation *root) : module_(root->getParentOfType<ModuleOp>())
{
    DBG_PRINT("ctor enter, root=%p, module_=%p\n", (void*)root, (void*)module_.getOperation());
    if (!module_) {
        DBG_PRINT("FATAL: module_ is null!\n");
        return;
    }
    OpBuilder builder(module_.getContext());
    DBG_PRINT("ctx=%p, builder=%p\n", (void*)module_.getContext(), (void*)&builder);
    for (auto type : {DepType::IntraCore, DepType::InterCore, DepType::LoadStore}) {
        DBG_PRINT("checking type=%d\n", (int)type);
        if (module_->getAttrOfType<IntegerAttr>(getAttrName(type))) {
            DBG_PRINT("  type=%d already set, skip\n", (int)type);
            continue;
        }
        DBG_PRINT("  type=%d not set, writing default\n", (int)type);
        module_->setAttr(getAttrName(type), builder.getI32IntegerAttr(getDefaultCount(type)));
    }
    DBG_PRINT("ctor exit\n");
}

void BufferCountManager::setBufferCount(DepType type, int count)
{
    DBG_PRINT("setBufferCount enter, type=%d count=%d module_=%p\n",
              (int)type, count, (void*)module_.getOperation());
    if (!module_) {
        DBG_PRINT("FATAL setBufferCount: module_ is null!\n");
        return;
    }
    if (count <= 0) {
        LOG_DEBUG("Invalid buffer count: " << count << " (must be > 0)");
        return;
    }
    if (count >= kBufferCountWarningThreshold) {
        LOG_DEBUG("Warning: buffer count " << count << " >= "
            << kBufferCountWarningThreshold << " is not recommended");
    }
    OpBuilder builder(module_.getContext());
    switch (type) {
        case DepType::IntraCore:
            module_->setAttr(CVPipeline::kIntraBufCount, builder.getI32IntegerAttr(count));
            LOG_DEBUG("IntraBufferCount set to " << count);
            break;
        case DepType::InterCore:
            module_->setAttr(CVPipeline::kInterCoreBufCount, builder.getI32IntegerAttr(count));
            LOG_DEBUG("InterBufferCount set to " << count);
            break;
        case DepType::LoadStore:
            module_->setAttr(CVPipeline::kLoadStoreBufCount, builder.getI32IntegerAttr(count));
            LOG_DEBUG("LoadBufferCount set to " << count);
            break;
        default:
            LOG_DEBUG("Unknown DepType: " << static_cast<int>(type));
            break;
    }
    DBG_PRINT("setBufferCount exit\n");
}

void BufferCountManager::buildBufferCountMap(
    llvm::DenseMap<Value, std::vector<Value>> &depValueMap,
    llvm::DenseMap<Value, int> &bufferCountMap,
    DepType type)
{
    int bufCount = getBufferCountByType(type);

    for (auto &p : depValueMap) {
        for (Value depVal : p.second) {
            if (isa<BlockArgument>(depVal) || !depVal.getDefiningOp())
                continue;
            bufferCountMap[depVal] = bufCount;
        }
    }
}

int BufferCountManager::getBufferCountByType(DepType type) const
{
    DBG_PRINT("getBufferCountByType enter, type=%d module_=%p\n",
              (int)type, (void*)module_.getOperation());
    auto attr = module_->getAttrOfType<IntegerAttr>(getAttrName(type));
    int count = static_cast<int>(attr.getInt());
    LOG_DEBUG("getBufferCountByType(" << static_cast<int>(type) << ") = " << count);
    return count;
}

} // namespace triton
} // namespace mlir