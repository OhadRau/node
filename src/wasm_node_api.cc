#include <type_traits>
#include <functional>
#include <uv.h>
#include "node_api.h"
#include "wasm/c-api.h"

wasm::Memory *memory;
using callback = auto (*)(const wasm::Val[], wasm::Val[]) -> wasm::own<wasm::Trap *>;

template<typename T>
constexpr auto wasmType = wasm::ValKind::I32;

template<>
constexpr auto wasmType<float> = wasm::ValKind::F32;

template<>
constexpr auto wasmType<double> = wasm::ValKind::F64;

template<wasm::ValKind... Args>
constexpr auto exportVoid(wasm::Store *store, callback cb) -> wasm::Extern * {
  return wasm::Func::make(
    store,
    wasm::FuncType::make(
      wasm::vec<wasm::ValType *>::make(
        wasm::ValType::make(Args).get()...
      ),
      wasm::vec<wasm::ValType *>::make()
    ).get(),
    cb
  ).get();
};

template<wasm::ValKind Return, wasm::ValKind... Args>
constexpr auto exportFn(wasm::Store *store, callback cb) -> wasm::Extern * {
  return wasm::Func::make(
    store,
    wasm::FuncType::make(
      wasm::vec<wasm::ValType *>::make(
        wasm::ValType::make(Args).get()...
      ),
      wasm::vec<wasm::ValType *>::make(
        wasm::ValType::make(Return).get()
      )
    ).get(),
    cb
  ).get();
};

template<typename Return, typename... Args>
constexpr auto exportNative(wasm::Store *store, callback cb) -> wasm::Extern * {
  if constexpr(std::is_void_v<Return>) {
    return exportVoid<wasmType<Args>...>(store, cb);
  } else {
    return exportFn<wasmType<Return>, wasmType<Args>...>(store, cb);
  }
}

template<typename T>
class Native {
public:
  using type = T;
};

template<typename T>
class Wasm {
public:
  using type = T;
};

template<typename>
struct is_native : public std::false_type {};

template<typename T>
struct is_native<Native<T>> : public std::true_type {};

template <typename T>
constexpr auto is_native_v = is_native<T>::value;

template<typename>
struct is_wasm : public std::false_type {};

template<typename T>
struct is_wasm<Wasm<T>> : public std::true_type {};

template <typename T>
constexpr auto is_wasm_v = is_wasm<T>::value;

template<class K>
using kind = typename K::type;

template<typename Arg>
constexpr Arg napiCast(const wasm::Val& v) {
  if constexpr(std::is_same_v<Arg, float>) {
    return (Arg) v.f32();
  } else if constexpr(std::is_same_v<Arg, double>) {
    return (Arg) v.f64();
  } else {
    return (Arg) v.i32();
  }
}

template<typename Result>
constexpr kind<Result> napiUnwrap(const wasm::Val& v) {
  if constexpr(is_wasm_v<Result>) {
    return (kind<Result>) (&memory->data()[v.i32()]);
  } else {
    return napiCast<kind<Result>>(v);
  }
};

template<typename Result>
constexpr wasm::Val napiWrap(kind<Result> v) {
  if constexpr(std::is_same_v<kind<Result>, float>) {
    return wasm::Val::f32((float) v);
  } else if (std::is_same_v<kind<Result>, double>) {
    return wasm::Val::f64((double) v);
  } else {
    return wasm::Val::i32((int) v);
  }
}

template<typename Result, typename... Args>
struct napiApply {
  template<size_t... Indices>
  static constexpr auto apply(
    kind<Result> (*fn)(kind<Args>...),
    const wasm::Val args[],
    std::index_sequence<Indices...>
  ) -> kind<Result> {
    if constexpr(std::is_void_v<kind<Result>>) {
      return fn(napiUnwrap<Args>(args[Indices])...);
    } else {
      fn(napiUnwrap<Args>(args[Indices])...);
    }
  }
};

template<typename Result, typename... Args>
struct napiCall {
private:
  using fp = kind<Result> (*)(kind<Args>...);
public:
  template<fp Fn>
  static wasm::own<wasm::Trap *> const call(
    const wasm::Val args[], wasm::Val results[]
  ) {
    if constexpr(std::is_void_v<Result>) {
      results[0] = napiWrap<Result>(
        napiApply<Result, Args...>::apply(Fn, args, std::index_sequence_for<Args...>{})
      );
    } else {
      napiApply<Result, Args...>::apply(Fn, args, std::index_sequence_for<Args...>{});
    }
    return nullptr;
  }
};

template<typename Result, typename... Args>
struct napiBind {
  template<kind<Result> (*const Fn)(kind<Args>...)>
  static auto bind(wasm::Store *store) -> wasm::Extern * {
    return exportNative<kind<Result>, kind<Args>...>(store, (callback) napiCall<Result, Args...>::template call<Fn>);
  }
};

void finalizeBuffer(napi_env env, void *finalize_data, void *finalize_hint) {
  free(finalize_data);
}

// napi_status napi_create_arraybuffer(napi_env env, size_t byte_length, void **data, napi_value *result)
// We need: void *data, void (*finalize)(napi_env, void *, void *)
wasm::own<wasm::Trap *> __napi_create_arraybuffer(
  const wasm::Val args[], wasm::Val results[]
) {
  napi_status status;

  napi_env env = (napi_env) args[0].i32();
  size_t byte_length = (size_t) args[1].i32();
  void **data = (void **) &memory->data()[args[2].i32()];
  napi_value *result = (napi_value *) &memory->data()[args[3].i32()];

  void *buffer = malloc(byte_length); // FIXME: malloc inside of Wasm
  if (!buffer) {
    status = napi_generic_failure;
  } else {
    status = napi_create_external_arraybuffer(env, buffer, byte_length, finalizeBuffer, nullptr, result);
    if (status == napi_ok) {
      *data = buffer;
    }
  }

  results[0] = wasm::Val::i32(status);
  return nullptr;
}

// napi_status napi_create_buffer(napi_env env, size_t size, void **data, napi_value *result)
// We need: void *data, void (*finalize)(napi_env, void *, void *)
wasm::own<wasm::Trap *> __napi_create_buffer(
  const wasm::Val args[], wasm::Val results[]
) {
  napi_status status;

  napi_env env = (napi_env) args[0].i32();
  size_t size = (size_t) args[1].i32();
  void **data = (void **) &memory->data()[args[2].i32()];
  napi_value *result = (napi_value *) &memory->data()[args[3].i32()];

  void *buffer = malloc(size); // FIXME: malloc, finalizer inside of Wasm
  if (!buffer) {
    status = napi_generic_failure;
  } else {
    status = napi_create_external_buffer(env, size, buffer, finalizeBuffer, nullptr, result);
    if (status == napi_ok) {
      *data = buffer;
    }
  }

  results[0] = wasm::Val::i32(status);
  return nullptr;
}

// napi_status napi_create_buffer_copy(napi_env env, size_t length, const void *data, void **result_data, napi_value *result)
wasm::own<wasm::Trap *> __napi_create_buffer_copy(
  const wasm::Val args[], wasm::Val results[]
) {
  napi_status status;

  napi_env env = (napi_env) args[0].i32();
  size_t length = (size_t) args[1].i32();
  void *data = (void *) &memory->data()[args[2].i32()];
  void **result_data = (void **) &memory->data()[args[3].i32()];
  napi_value *result = (napi_value *) &memory->data()[args[4].i32()];

  void *buffer = malloc(length); // FIXME: malloc, finalizer inside of Wasm
  if (!buffer) {
    status = napi_generic_failure;
  } else {
    status = napi_create_external_buffer(env, length, buffer, finalizeBuffer, nullptr, result);
    if (status == napi_ok) {
      memcpy(buffer, data, length);
      *result_data = buffer;
    }
  }

  results[0] = wasm::Val::i32(status);
  return nullptr;
}

// napi_status napi_get_arraybuffer_info(napi_env env, napi_value arraybuffer, void **data, size_t *byte_length)
wasm::own<wasm::Trap *> __napi_get_arraybuffer_info(
  const wasm::Val args[], wasm::Val results[]
) {
  napi_status status;

  napi_env env = (napi_env) args[0].i32();
  napi_value arraybuffer = (napi_value) args[1].i32();
  void **data = (void **) &memory->data()[args[2].i32()];
  size_t *byte_length = (size_t *) &memory->data()[args[3].i32()];

  // Warning: We assume that the ArrayBuffer was created using this API & the buffer is accessible to WASM
  status = napi_get_arraybuffer_info(env, arraybuffer, data, byte_length);
  if (status == napi_ok) {
    // Is there a smarter way to do this arithmetic?
    char *data_addr = (char *) *data;
    data_addr = (char *) (data_addr - (char *) &memory->data()[0]);
    *data = (void *) data_addr;
  }

  results[0] = wasm::Val::i32(status);
  return nullptr;
}

// napi_status napi_get_buffer_info(napi_env env, napi_value buffer, void **data, size_t *length)
wasm::own<wasm::Trap *> __napi_get_buffer_info(
  const wasm::Val args[], wasm::Val results[]
) {
  napi_status status;

  napi_env env = (napi_env) args[0].i32();
  napi_value arraybuffer = (napi_value) args[1].i32();
  void **data = (void **) &memory->data()[args[2].i32()];
  size_t *length = (size_t *) &memory->data()[args[3].i32()];

  // Warning: We assume that the ArrayBuffer was created using this API & the buffer is accessible to WASM
  status = napi_get_buffer_info(env, arraybuffer, data, length);
  if (status == napi_ok) {
    // Is there a smarter way to do this arithmetic?
    char *data_addr = (char *) *data;
    data_addr = (char *) (data_addr - (char *) &memory->data()[0]);
    *data = (void *) data_addr;
  }

  results[0] = wasm::Val::i32(status);
  return nullptr;
}

// TODO: napi_get_{typedarray,dataview}_info
// Since these aren't created by the WASM C API, their pointers won't be available for us...

auto provideNapi(
  wasm::Engine *engine, wasm::Module *module
) -> wasm::own<wasm::Instance *> {
  auto store = wasm::Store::make(engine).get();
  memory = wasm::Memory::make(
    store,
    wasm::MemoryType::make(wasm::Limits(0)).get()
  ).get();

  wasm::Extern *imports[] = {
    memory,
    // TODO: Define the napi_extended_error_info struct
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const napi_extended_error_info **>>
      ::bind<napi_get_last_error_info>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>>::bind<napi_throw>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char*>, Wasm<const char *>>
      ::bind<napi_throw_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char*>, Wasm<const char *>>
      ::bind<napi_throw_type_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char*>, Wasm<const char *>>
      ::bind<napi_throw_range_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_is_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_create_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_create_type_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_create_range_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_get_and_clear_last_exception>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<bool *>>::bind<napi_is_exception_pending>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>>::bind<napi_fatal_exception>(store),
    napiBind<Native<void>, Wasm<const char *>, Native<size_t>, Wasm<const char *>, Native<size_t>>
      ::bind<napi_fatal_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_handle_scope *>>::bind<napi_open_handle_scope>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_handle_scope>>::bind<napi_close_handle_scope>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_escapable_handle_scope *>>
      ::bind<napi_open_escapable_handle_scope>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_escapable_handle_scope>>
      ::bind<napi_close_escapable_handle_scope>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_escapable_handle_scope>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_escape_handle>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<uint32_t>, Wasm<napi_ref *>>
      ::bind<napi_create_reference>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_ref>>::bind<napi_delete_reference>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_ref>, Wasm<uint32_t *>>::bind<napi_reference_ref>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_ref>, Wasm<uint32_t *>>::bind<napi_reference_unref>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_ref>, Wasm<napi_value *>>::bind<napi_get_reference_value>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<void (*)(void *)>, Wasm<void *>>::bind<napi_add_env_cleanup_hook>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<void (*)(void *)>, Wasm<void *>>::bind<napi_remove_env_cleanup_hook>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_create_array>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<size_t>, Wasm<napi_value *>>::bind<napi_create_array_with_length>(store),
    exportFn<wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32>(store, __napi_create_arraybuffer),
    exportFn<wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32>(store, __napi_create_buffer),
    exportFn<wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32>(store, __napi_create_buffer_copy),
    #if NAPI_VERSION >= 4 && NAPI_EXPERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<double>, Wasm<napi_value *>>::bind<napi_create_date>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<void *>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_value *>>
      ::bind<napi_create_external>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<void *>, Native<size_t>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_value *>>
      ::bind<napi_create_external_arraybuffer>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<size_t>, Wasm<void *>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_value *>>
      ::bind<napi_create_external_buffer>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_create_object>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_create_symbol>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_typedarray_type>, Native<size_t>, Native<napi_value>, Native<size_t>, Wasm<napi_value *>>
      ::bind<napi_create_typedarray>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<size_t>, Native<napi_value>, Native<size_t>, Wasm<napi_value *>>
      ::bind<napi_create_dataview>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<int32_t>, Wasm<napi_value *>>::bind<napi_create_int32>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<uint32_t>, Wasm<napi_value *>>::bind<napi_create_uint32>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<int64_t>, Wasm<napi_value *>>::bind<napi_create_int64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<double>, Wasm<napi_value *>>::bind<napi_create_double>(store),
    #if NAPI_VERSION >= 4 && NAPI_EXPERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<int64_t>, Wasm<napi_value *>>::bind<napi_create_bigint_int64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<uint64_t>, Wasm<napi_value *>>::bind<napi_create_bigint_uint64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<int>, Native<size_t>, Wasm<const int64_t *>, Wasm<napi_value *>>::bind<napi_create_bigint_words>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char *>, Native<size_t>, Wasm<napi_value *>>::bind<napi_create_string_latin1>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char16_t *>, Native<size_t>, Wasm<napi_value *>>::bind<napi_create_string_utf16>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char *>, Native<size_t>, Wasm<napi_value *>>::bind<napi_create_string_utf8>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<uint32_t *>>::bind<napi_get_array_length>(store),
    exportFn<wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32>(store, __napi_get_arraybuffer_info),
    exportFn<wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32, wasm::ValKind::I32>(store, __napi_get_buffer_info),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_get_prototype>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_typedarray_type *>, Wasm<size_t *>, Wasm<void **>, Wasm<napi_value *>, Wasm<size_t *>>
      ::bind<napi_get_typedarray_info>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<size_t *>, Wasm<void **>, Wasm<napi_value *>, Wasm<size_t *>>
      ::bind<napi_get_dataview_info>(store),
    #if NAPI_VERSION >= 4 && NAPI_EXERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<double *>>::bind<napi_get_date_value>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_get_value_bool>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<double *>>::bind<napi_get_value_double>(store),
    #if NAPI_VERSION >= 4 && NAPI_EXPERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<int64_t *>, Wasm<bool *>>
      ::bind<napi_get_value_bigint_int64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<uint64_t *>, Wasm<bool *>>
      ::bind<napi_get_value_bigint_uint64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<int *>, Wasm<size_t *>, Wasm<uint64_t *>>
      ::bind<napi_get_value_bigint_words>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<void **>>::bind<napi_get_value_external>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<int32_t *>>::bind<napi_get_value_int32>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<uint32_t *>>::bind<napi_get_value_uint32>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<int64_t *>>::bind<napi_get_value_int64>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<char *>, Native<size_t>, Wasm<size_t *>>
      ::bind<napi_get_value_string_latin1>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<char *>, Native<size_t>, Wasm<size_t *>>
      ::bind<napi_get_value_string_utf8>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<char16_t *>, Native<size_t>, Wasm<size_t *>>
      ::bind<napi_get_value_string_utf16>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<bool>, Wasm<napi_value *>>::bind<napi_get_boolean>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_get_global>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_get_null>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_value *>>::bind<napi_get_undefined>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_coerce_to_bool>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_coerce_to_number>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_coerce_to_object>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>::bind<napi_coerce_to_string>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_valuetype *>>::bind<napi_typeof>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_instanceof>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_array>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_arraybuffer>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_buffer>(store),
    #if NAPI_VERSION >= 4 && NAPI_EXPERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_date>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_error>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_typedarray>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>::bind<napi_is_dataview>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_strict_equals>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_get_property_names>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Native<napi_value>>
      ::bind<napi_set_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_get_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_has_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_delete_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_has_own_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<const char *>, Native<napi_value>>
      ::bind<napi_set_named_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<const char *>, Wasm<napi_value *>>
      ::bind<napi_get_named_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<const char *>, Wasm<bool *>>
      ::bind<napi_has_named_property>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<uint32_t>, Native<napi_value>>
      ::bind<napi_set_element>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<uint32_t>, Wasm<napi_value *>>
      ::bind<napi_get_element>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<uint32_t>, Wasm<bool *>>
      ::bind<napi_has_element>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<uint32_t>, Wasm<bool *>>
      ::bind<napi_delete_element>(store),
    // TODO: Copy the property descriptor struct
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<size_t>, Wasm<const napi_property_descriptor *>>
      ::bind<napi_define_properties>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Native<size_t>, Wasm<const napi_value *>, Wasm<napi_value *>>
      ::bind<napi_call_function>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char *>, Native<size_t>, Wasm<napi_callback>, Wasm<void *>, Wasm<napi_value *>>
      ::bind<napi_create_function>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_callback_info>, Wasm<size_t *>, Wasm<napi_value *>, Wasm<napi_value *>, Wasm<void **>>
      ::bind<napi_get_cb_info>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_callback_info>, Wasm<napi_value *>>
      ::bind<napi_get_new_target>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<size_t>, Wasm<const napi_value *>, Wasm<napi_value *>>
      ::bind<napi_new_instance>(store),
    // TODO: Copy the property descriptor struct
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const char *>, Native<size_t>, Wasm<napi_callback>, Wasm<void *>, Native<size_t>, Wasm<const napi_property_descriptor *>, Wasm<napi_value *>>
      ::bind<napi_define_class>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<void *>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_ref *>>
      ::bind<napi_wrap>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<void **>>::bind<napi_unwrap>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<void **>>::bind<napi_remove_wrap>(store),
    #if NAPI_VERSION >= 4 && NAPI_EXPERIMENTAL
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<void *>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_ref *>>
      ::bind<napi_add_finalizer>(store),
    #endif
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_async_execute_callback>, Wasm<napi_async_complete_callback>, Wasm<void *>, Wasm<napi_async_work *>>
      ::bind<napi_create_async_work>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_async_work>>::bind<napi_delete_async_work>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_async_work>>::bind<napi_queue_async_work>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_async_work>>::bind<napi_cancel_async_work>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Wasm<napi_async_context *>>
      ::bind<napi_async_init>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_async_context>>
      ::bind<napi_async_destroy>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_async_context>, Native<napi_value>, Native<napi_value>, Native<size_t>, Wasm<const napi_value *>, Wasm<napi_value *>>
      ::bind<napi_make_callback>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_async_context>, Wasm<napi_callback_scope *>>
      ::bind<napi_open_callback_scope>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_callback_scope>>::bind<napi_close_callback_scope>(store),
    // TODO: Copy the node version struct
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<const napi_node_version **>>::bind<napi_get_node_version>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<uint32_t *>>::bind<napi_get_version>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<int64_t>, Wasm<int64_t *>>::bind<napi_adjust_external_memory>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<napi_deferred *>, Wasm<napi_value *>>
      ::bind<napi_create_promise>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_deferred>, Native<napi_value>>
      ::bind<napi_resolve_deferred>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_deferred>, Native<napi_value>>
      ::bind<napi_reject_deferred>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<bool *>>
      ::bind<napi_is_promise>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Wasm<napi_value *>>
      ::bind<napi_run_script>(store),
    // TODO: What kind of copying has to be done here?
    napiBind<Native<napi_status>, Native<napi_env>, Wasm<uv_loop_t **>>
      ::bind<napi_get_uv_event_loop>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_value>, Native<napi_value>, Native<napi_value>, Native<size_t>, Native<size_t>, Wasm<void *>, Wasm<napi_finalize>, Wasm<void *>, Wasm<napi_threadsafe_function_call_js>, Wasm<napi_threadsafe_function *>>
      ::bind<napi_create_threadsafe_function>(store),
    napiBind<Native<napi_status>, Native<napi_threadsafe_function>, Wasm<void **>>
      ::bind<napi_get_threadsafe_function_context>(store),
    napiBind<Native<napi_status>, Native<napi_threadsafe_function>, Wasm<void *>, Native<napi_threadsafe_function_call_mode>>
      ::bind<napi_call_threadsafe_function>(store),
    napiBind<Native<napi_status>, Native<napi_threadsafe_function>>
      ::bind<napi_acquire_threadsafe_function>(store),
    napiBind<Native<napi_status>, Native<napi_threadsafe_function>, Native<napi_threadsafe_function_release_mode>>
      ::bind<napi_release_threadsafe_function>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_threadsafe_function>>
      ::bind<napi_ref_threadsafe_function>(store),
    napiBind<Native<napi_status>, Native<napi_env>, Native<napi_threadsafe_function>>
      ::bind<napi_unref_threadsafe_function>(store),
  };
  auto instance = wasm::Instance::make(
    store,
    module,
    imports
  );
  return instance;
}
