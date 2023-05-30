#pragma once

#include <cstddef>
#include <vector>

#include "debug.h"
#include "memory/memory_ref.h"

template <typename ElemType> struct CircularBufferState {
  size_t head;
  size_t tail;
  size_t length;
  size_t capacity;
  gear::memory::CpuMemoryRef data;

  CircularBufferState(size_t head, size_t tail, size_t length, size_t capacity,
                      gear::memory::Memory &&data_mem)
      : head(head), tail(tail), length(length), capacity(capacity),
        data(
            std::make_shared<gear::memory::CpuMemoryRef>(std::move(data_mem))) {
  }

  CircularBufferState(size_t head, size_t tail, size_t length, size_t capacity,
                      const ElemType *dptr)
      : head(head), tail(tail), length(length), capacity(capacity),
        data(reinterpret_cast<const void *>(dptr), 0,
             capacity * sizeof(ElemType)) {}

  CircularBufferState(const CircularBufferState<ElemType> &state)
      : head(state.head), tail(state.tail), length(state.length),
        capacity(state.capacity), data(state.data) {}

  CircularBufferState(CircularBufferState<ElemType> &&state)
      : head(state.head), tail(state.tail), length(state.length),
        capacity(state.capacity), data(std::move(data)) {}
};

template <typename ElemType> class CircularBuffer {
public:
  CircularBuffer(size_t capacity);

  CircularBufferState<ElemType> get_state() const;

  void set_state(const CircularBufferState<ElemType> &state);

  bool push(const ElemType &elem);

  bool pop(ElemType &elem);

  size_t size();

  template <typename... Args> bool emplace(Args &&...args);

private:
  size_t head = 0;
  size_t tail = 0;
  size_t length = 0;
  size_t capacity;
  std::vector<ElemType> data;
};

template <typename ElemType>
CircularBuffer<ElemType>::CircularBuffer(size_t capacity) {
  this->capacity = capacity;
  this->data.reserve(capacity);
}

template <typename ElemType>
CircularBufferState<ElemType> CircularBuffer<ElemType>::get_state() const {
  GEAR_DEBUG_PRINT("IndexBuffer enter get state call");
  return CircularBufferState<ElemType>(this->head, this->tail, this->length,
                                       this->capacity, this->data.data());
}

template <typename ElemType>
void CircularBuffer<ElemType>::set_state(
    const CircularBufferState<ElemType> &state) {
  this->data.reserve(state.capacity);
  this->head = state.head;
  this->tail = state.tail;
  this->length = state.length;
  this->capacity = state.capacity;
  memcpy(reinterpret_cast<void *>(const_cast<ElemType *>(this->data.data())),
         state.data.raw->addr, sizeof(ElemType) * this->capacity);
}

template <typename ElemType>
bool CircularBuffer<ElemType>::push(const ElemType &elem) {
  if (this->length == this->capacity) {
    return false;
  }

  data[tail] = elem;
  tail = (tail + 1) % this->capacity;
  ++length;
  return true;
}

template <typename ElemType>
bool CircularBuffer<ElemType>::pop(ElemType &elem) {
  if (this->length == 0) {
    return false;
  }
  elem = data[head];
  head = (head + 1) % this->capacity;
  --length;
  return true;
}

template <typename ElemType> size_t CircularBuffer<ElemType>::size() {
  return length;
}

template <typename ElemType>
template <typename... Args>
bool CircularBuffer<ElemType>::emplace(Args &&...args) {
  data[tail] = std::move(ElemType(std::forward<Args>(args)...));
  tail = (tail + 1) % this->capacity;
  ++length;
  return true;
}