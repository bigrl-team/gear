#pragma once

#include "common/cmp.h"

namespace gear::common {

struct Range {
  size_t lb;

  size_t hb;

  Range(size_t low, size_t high) : lb(low), hb(high) {}

  Range(const Range &r) : lb(r.lb), hb(r.hb) {}

  Range join(const Range &other) const {
    size_t nlb = MAX(this->lb, other.lb);
    size_t nhb = MIN(this->hb, other.hb);
    return Range(nlb, nhb);
  }

  Range merge(const Range &other) const {
    Range joined = this->join(other);
    if (joined.valid()) {
      return Range(MIN(this->lb, other.lb), MAX(this->hb, other.hb));
    } else {
      return joined;
    }
  }

  bool valid() { return lb <= hb; }

  size_t size() { return hb - lb; }
};

} // namespace gear::common