# DART Callback Mechanism - Step-by-Step Explanation

This document explains how the DART-specific callback mechanism works in LightGBM, from Python setup through C++ execution.

## Overview

The DART callback mechanism allows Python code to be called during DART training iterations. The callback is invoked at two critical points:
1. **After a new tree is trained** (before normalization) - for SHAP computation
2. **During DroppingTrees()** - to allow setting drop indices via `set_dart_drop_indices()`

---

## Phase 1: Setting Up the Callback (Initialization)

### Step 1: Python User Calls `set_dart_callback()`

```python
booster.set_dart_callback(my_callback, user_data=booster)
```

### Step 2: Python Wrapper Creation (`basic.py:4209-4214`)

- Creates a wrapper function that captures `user_data` in a closure
- Converts the Python function to a C callback using `_DART_CB` (ctypes)
- Stores the C callback in `self._dart_cb` to prevent garbage collection

**Code:**
```python
def callback_wrapper(iter, _):
    callback(iter, user_data)

c_cb = _DART_CB(callback_wrapper)
self._dart_cb = c_cb  # Store to prevent GC
```

### Step 3: C API Call (`basic.py:4219-4224`)

- Calls `LGBM_BoosterSetDartCallback()` with:
  - Booster handle
  - Callback function pointer (cast to `void*`)
  - `user_data` (passed via closure, so `NULL` here)

**Code:**
```python
callback_ptr = ctypes.cast(c_cb, ctypes.c_void_p)
_LIB.LGBM_BoosterSetDartCallback(
    self._handle,
    callback_ptr,
    ctypes.c_void_p(0),  # user_data passed via closure
)
```

### Step 4: C++ Booster Stores Callback (`c_api.cpp:3046-3052`)

- `LGBM_BoosterSetDartCallback()` calls `Booster::SetDartCallback()`
- Stores callback in `dart_cb_` and `dart_cb_data_` member variables

**Code:**
```cpp
int LGBM_BoosterSetDartCallback(BoosterHandle handle,
                                LGBMDartCallback callback,
                                void* user_data) {
  Booster* ref_booster = reinterpret_cast<Booster*>(handle);
  ref_booster->SetDartCallback(callback, user_data);
}

void SetDartCallback(LGBMDartCallback cb, void* user_data) {
  dart_cb_ = cb;
  dart_cb_data_ = user_data;
}
```

---

## Phase 2: Training Iteration Flow

### Step 5: User Calls `booster.update()`

- This calls `LGBM_BoosterUpdateOneIter()` which calls `Booster::TrainOneIter()`

### Step 6: Thread-Local Storage Setup (`c_api.cpp:424-428`)

Before calling `boosting_->TrainOneIter()`, the Booster class:
- Saves the current thread-local callback (if any)
- Sets `g_dart_callback = dart_cb_`
- Sets `g_dart_callback_data = dart_cb_data_`

This makes the callback accessible to DART code via thread-local storage.

**Code:**
```cpp
bool TrainOneIter() {
  UNIQUE_LOCK(mutex_)
  // Set thread-local callback for DART to access
  LGBMDartCallback old_cb = LightGBM::g_dart_callback;
  void* old_data = LightGBM::g_dart_callback_data;
  LightGBM::g_dart_callback = dart_cb_;
  LightGBM::g_dart_callback_data = dart_cb_data_;
  
  bool result = boosting_->TrainOneIter(nullptr, nullptr);
  
  // Restore old callback (if any)
  LightGBM::g_dart_callback = old_cb;
  LightGBM::g_dart_callback_data = old_data;
  
  return result;
}
```

### Step 7: DART Training Begins (`dart.hpp:71-76`)

- `DART::TrainOneIter()` is called
- Calls `GBDT::TrainOneIter()` to train a new tree
- After the tree is trained, but before normalization...

### Step 8: First Callback Invocation - SHAP Computation Phase (`dart.hpp:81-83`)

```cpp
if (g_dart_callback != nullptr) {
  g_dart_callback(iter_, g_dart_callback_data);
}
```

**What happens:**
- The callback is called with the current iteration number
- Trees are still in a valid state (before normalization)
- Python callback can:
  - Compute SHAP values for the new tree
  - Access tree structure
  - Prepare drop decisions

**Code location:** `dart.hpp:78-83`

### Step 9: Normalization Happens (`dart.hpp:86`)

- DART normalizes the tree weights
- This modifies the tree structure

### Step 10: Training Score is Requested

- When `GetTrainingScore()` is called, it triggers `DroppingTrees()`

**Code:**
```cpp
const double* GetTrainingScore(int64_t* out_len) override {
  if (!is_update_score_cur_iter_) {
    DroppingTrees();  // <-- Triggers drop decision phase
    is_update_score_cur_iter_ = true;
  }
  // ...
}
```

### Step 11: Second Callback Invocation - Drop Decision Phase (`dart.hpp:120-133`)

```cpp
void DroppingTrees() {
  drop_index_.clear();
  
  // Clear drop indices storage
  if (g_dart_drop_indices != nullptr) {
    g_dart_drop_indices->clear();
  }
  
  // Call callback - THIS IS WHERE DROP DECISIONS ARE MADE
  if (g_dart_callback != nullptr) {
    g_dart_callback(iter_, g_dart_callback_data);
  }
  
  // Check if callback set drop indices...
}
```

**What happens:**
- The callback is called again with the same iteration number
- Python callback can now call `set_dart_drop_indices()`
- This is the drop decision phase

---

## Phase 3: Setting Drop Indices (From Python Callback)

### Step 12: Python Callback Calls `set_dart_drop_indices()`

```python
def my_callback(iteration, userdata):
    booster = userdata
    # ... compute which trees to drop ...
    booster.set_dart_drop_indices([0, 1, 5])  # Drop trees 0, 1, and 5
```

### Step 13: Python Converts to C Array (`basic.py:4247-4252`)

- Converts Python list to C integer array
- Calls `LGBM_DartSetDropIndices()`

**Code:**
```python
indices_array = (ctypes.c_int * len(indices))(*indices)
_LIB.LGBM_DartSetDropIndices(indices_array, len(indices))
```

### Step 14: C API Stores in Thread-Local Vector (`c_api.cpp:3055-3069`)

```cpp
int LGBM_DartSetDropIndices(const int* indices, int num_indices) {
  // Use static thread-local storage for the vector
  static thread_local std::vector<int> drop_indices_storage;
  LightGBM::g_dart_drop_indices = &drop_indices_storage;
  
  // Clear and populate with new indices
  LightGBM::g_dart_drop_indices->clear();
  if (indices != nullptr && num_indices > 0) {
    LightGBM::g_dart_drop_indices->reserve(num_indices);
    for (int i = 0; i < num_indices; ++i) {
      LightGBM::g_dart_drop_indices->push_back(indices[i]);
    }
  }
}
```

**What happens:**
- Stores drop indices in thread-local `g_dart_drop_indices` vector
- This is the same thread that's running DART training

### Step 15: DART Checks for Drop Indices (`dart.hpp:135-138`)

```cpp
if (g_dart_drop_indices != nullptr && !g_dart_drop_indices->empty()) {
  // Use drop indices from Python callback
  drop_index_ = *g_dart_drop_indices;
} else {
  // Fall back to original random selection logic
  bool is_skip = random_for_drop_.NextFloat() < config_->skip_drop;
  // ... random drop selection based on drop_rate ...
}
```

**What happens:**
- If the callback set drop indices, use them
- Otherwise, fall back to random selection based on `drop_rate`

### Step 16: Trees are Dropped (`dart.hpp:160+`)

- DART applies the drop indices
- Shrinks the selected trees
- Updates scores accordingly

### Step 17: Thread-Local Storage Restored (`c_api.cpp:432-434`)

- After `TrainOneIter()` completes, restore the old callback
- This allows nested callbacks or different callbacks in different threads

**Code:**
```cpp
// Restore old callback (if any)
LightGBM::g_dart_callback = old_cb;
LightGBM::g_dart_callback_data = old_data;
```

---

## Key Design Points

### 1. Two Invocation Points

The callback is called twice per iteration:

- **After tree training (before normalization)**: 
  - Trees are in a valid state
  - Perfect for SHAP computation
  - Can access tree structure
  
- **During DroppingTrees()**: 
  - Can set drop indices
  - Override random selection
  - Make intelligent drop decisions

### 2. Thread-Local Storage

- Callback is stored in thread-local variables (`g_dart_callback`, `g_dart_callback_data`)
- Allows DART code to access it without passing parameters
- Thread-safe for parallel training
- Each thread has its own callback instance

### 3. Drop Indices Mechanism

- Python callback calls `set_dart_drop_indices()`
- Indices stored in thread-local vector (`g_dart_drop_indices`)
- DART checks this vector and uses it if set
- Falls back to random selection if not set

### 4. Callback Lifecycle

1. **Set once** via `set_dart_callback()`
2. **Stored** in Booster instance (`dart_cb_`, `dart_cb_data_`)
3. **Activated per iteration** via thread-local storage
4. **Called twice** per iteration (SHAP phase + drop phase)
5. **Restored** after each iteration

---

## Example Usage

```python
import lightgbm as lgb
import numpy as np

def my_dart_callback(iteration, userdata):
    """Callback function called during DART training."""
    booster = userdata
    
    print(f"DART iteration: {iteration}")
    print(f"Total trees: {booster.num_trees()}")
    
    # Compute which trees to drop based on custom logic
    # (e.g., using SHAP values, tree importance, etc.)
    trees_to_drop = [0, 1, 2]  # Example: drop first 3 trees
    
    # Set drop indices (only works when called from DroppingTrees phase)
    booster.set_dart_drop_indices(trees_to_drop)

# Create data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
train_data = lgb.Dataset(X_train, label=y_train)

# Create booster with DART
params = {
    'objective': 'regression',
    'boosting_type': 'dart',
    'drop_rate': 0.1,
    'num_iterations': 10,
    'verbose': -1
}

booster = lgb.Booster(params, train_set=train_data)

# Set callback
booster.set_dart_callback(my_dart_callback, user_data=booster)

# Train - callback will be triggered during each iteration
for i in range(5):
    booster.update()
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ booster.set_dart_callback(callback, user_data)      │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ctypes wrapper → C callback function                 │   │
│  └──────────────────┬───────────────────────────────────┘   │
└─────────────────────┼───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ C API Layer                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LGBM_BoosterSetDartCallback()                        │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Booster::SetDartCallback()                           │   │
│  │   - Stores in dart_cb_, dart_cb_data_                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Training Iteration                                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Booster::TrainOneIter()                              │   │
│  │   - Sets thread-local: g_dart_callback              │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ DART::TrainOneIter()                                │   │
│  │   - Trains new tree                                 │   │
│  │   - CALLBACK #1: After tree, before normalization │   │
│  │   - Normalizes tree                                 │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ DART::GetTrainingScore()                             │   │
│  │   - Calls DroppingTrees()                             │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ DART::DroppingTrees()                                 │   │
│  │   - CALLBACK #2: During drop decision                │   │
│  │   - Checks g_dart_drop_indices                        │   │
│  │   - Uses callback indices OR random selection        │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ Python Callback Execution                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ callback(iteration, user_data)                      │   │
│  │   - Can compute SHAP values                         │   │
│  │   - Can call booster.set_dart_drop_indices([...])  │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ LGBM_DartSetDropIndices()                            │   │
│  │   - Stores in thread-local g_dart_drop_indices       │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

This design allows Python code to:
- ✅ Compute SHAP values incrementally
- ✅ Make intelligent drop decisions
- ✅ Override random tree selection
- ✅ Access the booster state during training

The callback mechanism is **DART-specific** and only active when `boosting_type='dart'`. It provides a clean separation between the training logic and custom Python code, enabling advanced use cases like SHAP-based tree dropping.

