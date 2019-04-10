#include "memory_dag.h"

#include <torch/csrc/utils/memory.h>
#include <algorithm>
#include <queue>

namespace torch {
namespace jit {

bool MemoryDAG::mayAlias(Element* a, Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAlias(const Element* a, const Element* b) const {
  return mayAliasImpl(a, b);
}

bool MemoryDAG::mayAliasImpl(const Element* a, const Element* b) const {
  const auto aMemLoc = a->getMemoryLocations();
  const auto bMemLoc = b->getMemoryLocations();

  // XXX: This could be more efficiently done as a bitwise AND on two bitfields
  // that represent memory location membership. If these comparisons end up
  // being a bottleneck, consider implementing it that way.
  for (const auto aLoc : aMemLoc) {
    for (const auto bLoc : bMemLoc) {
      if (aLoc == bLoc) {
        return true;
      }
    }
  }
  return false;
}

bool MemoryDAG::mayContainAlias(
    const Element* contained,
    const Element* container) const {
  return mayContainAliasImpl(contained, container);
}

bool MemoryDAG::mayContainAlias(Element* contained, Element* container) const {
  return mayContainAliasImpl(contained, container);
}

bool MemoryDAG::mayContainAliasImpl(
    const Element* elem,
    const Element* container) const {
  // If elem contains other elements, does container hold any of those elements?

  if (std::any_of(
          elem->contained_elements.begin(),
          elem->contained_elements.end(),
          [&](const Element* e) {
            return mayContainAliasImpl(e, container);
          })) {
    return true;
  }

  // If container contains other elements, do those containers hold elem?
  if (std::any_of(
          container->contained_elements.begin(),
          container->contained_elements.end(),
          [&](const Element* cont) {
            return mayContainAliasImpl(elem, cont);
          })) {
    return true;
  }

  const auto elemMemLoc = elem->getMemoryLocations();
  const auto contMemLoc = container->getMemoryLocations();

  for (const auto elem_mem : elemMemLoc) {
    for (const auto container : contMemLoc) {
      const auto& cont_elem = container->contained_elements;
      if (std::find(cont_elem.begin(), cont_elem.end(), elem_mem) !=
          cont_elem.end()) {
        return true;
      };
    }
  }

  return false;
}

// Make `v` point at `to`.
void MemoryDAG::makePointerTo(Element* from, Element* to) {
  from->pointsTo.insert(to);
  to->pointedFrom.insert(from);
}

void MemoryDAG::addToContainedElements(Element* elem, Element* container) {
  container->contained_elements.push_back(elem);
}

// Give `v` a fresh alias (i.e. it does not point to any value)
Element* MemoryDAG::makeFreshValue(const Value* v) {
  auto el = torch::make_unique<Element>();
  el->value = v;

  auto rawPtr = el.get();
  elements_.emplace(rawPtr, std::move(el));
  return rawPtr;
}

std::unordered_set<const Element*> Element::getMemoryLocations() const {
  if (!cachedMemoryLocations_.empty()) {
    return cachedMemoryLocations_;
  }

  // Do a BFS in the `points-to` direction, collecting all memory locations
  std::unordered_set<const Element*> ret;
  this->bfs(
      [&](const Element* el) {
        if (el->pointsTo.empty()) {
          ret.insert(el);
        }
      },
      BfsDirection::POINTS_TO);

  cachedMemoryLocations_ = ret;
  return ret;
}

// Do a breadth-first search over the graph, starting at `this` and
// traversing in the direction `dir`.`fn` will be run on each element.
template <typename Fn>
bool Element::bfs(Fn fn, BfsDirection dir) const {
  std::queue<const Element*> queue;
  std::unordered_set<const Element*> seen;

  queue.push(this);
  while (!queue.empty()) {
    const auto el = queue.front();
    queue.pop();
    seen.insert(el);

    fn(el);

    switch (dir) {
      case BfsDirection::POINTS_TO: {
        for (auto ptr : el->pointsTo) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
      } break;

      case BfsDirection::POINTED_FROM: {
        for (auto ptr : el->pointedFrom) {
          if (!seen.count(ptr)) {
            queue.push(ptr);
          }
        }
      } break;
    }
  }
  return false;
}
} // namespace jit
} // namespace torch
