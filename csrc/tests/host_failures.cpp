// Failure paths of the ROCm layer's host code that a real GPU can't produce on demand: HIP calls
// failing during cleanup, and host memory running out for good (review round 119, R139 and R140).
// runtime.hip is compiled against the stand-in HIP of fake_hip/, and operator new is replaced
// so that a test can make every host allocation fail. Each mode runs in its own process, so that
// an abort (a C++ exception escaping a noexcept function) fails only that mode.
//
// Built and run by tools/check_rocm_host_failures.py; not part of the library.

// GCC takes the replaced operator delete's free for a mismatch with operator new.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#include "../runtime.hip"

void *operator new(std::size_t n) {
    if (fake::deny_new) throw std::bad_alloc();
    if (void *p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void *operator new[](std::size_t n) { return operator new(n); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { std::free(p); }
void operator delete(void *p, std::size_t) noexcept { std::free(p); }
void operator delete[](void *p, std::size_t) noexcept { std::free(p); }

namespace {

int failures = 0;

// Checks without allocating (some checks run while operator new throws).
#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

#define CHECK_ERROR(expected)                                                                      \
    do {                                                                                           \
        const char *actual_ = hmr_last_error();                                                    \
        if (std::strcmp(actual_, expected) != 0) {                                                 \
            std::fprintf(stderr, "%s:%d: last error is \"%s\", expected \"%s\"\n", __FILE__,      \
                         __LINE__, actual_, expected);                                             \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

// The streams HIP holds agree with what hmr_test_live_streams reports.
#define CHECK_STREAMS(n)                                                                           \
    do {                                                                                           \
        CHECK(fake::live_streams == (n));                                                          \
        CHECK(hmr_test_live_streams() == (n));                                                     \
    } while (0)

// A context can still be created and destroyed, leaving the thread on its device.
void create_and_destroy() {
    fake::launch_error = hipSuccess;
    fake::destroy_error = hipSuccess;
    int before = fake::live_streams;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OK);
    CHECK(ctx != nullptr);
    CHECK_STREAMS(before + 1);
    CHECK(hmr_context_destroy(ctx) == HMR_OK);
    CHECK_STREAMS(before);
    CHECK(fake::current_device == 1);
}

// Context creation fails after the stream exists: the stream is destroyed on device 0, before the
// thread is put back on device 1.
void late_failure() {
    hmr_test_fail_after_probe(1);
    hmr_context_t ctx = reinterpret_cast<hmr_context_t>(1);
    CHECK(hmr_context_create(0, &ctx) == HMR_OUT_OF_MEMORY);
    CHECK(ctx == nullptr);
    CHECK_ERROR("host allocation failed");
    CHECK_STREAMS(0);
    CHECK(fake::destroyed_on == 0);
    CHECK(fake::current_device == 1);
    create_and_destroy();
}

void probe_failure() {
    fake::launch_error = hipErrorOutOfMemory;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OUT_OF_MEMORY);
    CHECK(ctx == nullptr);
    CHECK_ERROR("rocm:0: running the probe kernel: out of memory");
    CHECK_STREAMS(0);
    CHECK(fake::destroyed_on == 0);
    CHECK(fake::current_device == 1);
    create_and_destroy();
}

// The probe fails and so does destroying the stream: the probe's error is the one reported, and
// the stream HIP still holds is counted (R139).
void probe_then_destroy_failure() {
    fake::launch_error = hipErrorOutOfMemory;
    fake::destroy_error = hipErrorInvalidHandle;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OUT_OF_MEMORY);
    CHECK(ctx == nullptr);
    CHECK_ERROR("rocm:0: running the probe kernel: out of memory");
    CHECK_STREAMS(1);
    CHECK(fake::current_device == 1);
    create_and_destroy();
}

// Destroying the stream fails while no host memory is left: no exception escapes the cleanup (it
// would abort), and the original error stands (R139).
void destroy_failure_without_memory() {
    hmr_test_fail_after_probe(1);
    fake::destroy_error = hipErrorInvalidHandle;
    fake::deny_new_after_destroy = true;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OUT_OF_MEMORY);
    CHECK(ctx == nullptr);
    CHECK_ERROR("host allocation failed");
    CHECK_STREAMS(1);
    CHECK(fake::current_device == 1);
    fake::deny_new_after_destroy = false;
    fake::deny_new = false;
    create_and_destroy();
}

// The explicit destroy reports its own failure, and counts the stream HIP kept.
void destroy_failure() {
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OK);
    fake::destroy_error = hipErrorInvalidHandle;
    CHECK(hmr_context_destroy(ctx) == HMR_HIP_ERROR);
    CHECK_ERROR("hipStreamDestroy(ctx->stream): invalid resource handle");
    CHECK_STREAMS(1);
    CHECK(fake::current_device == 1);
    create_and_destroy();
}

// The explicit destroy fails while no host memory is left: the status stays the HIP error's,
// with HIP's text for it (review round 120, R143).
void destroy_failure_without_memory_explicit() {
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OK);
    fake::destroy_error = hipErrorInvalidHandle;
    fake::deny_new_after_destroy = true;
    CHECK(hmr_context_destroy(ctx) == HMR_HIP_ERROR);
    CHECK_ERROR("invalid resource handle");
    CHECK_STREAMS(1);
    CHECK(fake::current_device == 1);
    fake::deny_new_after_destroy = false;
    fake::deny_new = false;
    create_and_destroy();
}

// The probe fails and host memory runs out from then on: neither the probe's message nor the
// context's longer one can be stored, and the status stays the HIP error's (R143).
void probe_failure_without_memory() {
    fake::launch_error = hipErrorInvalidHandle;
    fake::deny_new_after_launch = true;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_HIP_ERROR);
    CHECK(ctx == nullptr);
    CHECK_ERROR("invalid resource handle");
    CHECK_STREAMS(0);
    CHECK(fake::current_device == 1);
    fake::deny_new_after_launch = false;
    fake::deny_new = false;
    create_and_destroy();
}

// Host memory runs out for good: entry points return HMR_OUT_OF_MEMORY instead of aborting in
// guarded's handler (R140); once memory is back, messages are stored again.
void persistent_oom() {
    fake::deny_new = true;
    hmr_context_t ctx = nullptr;
    CHECK(hmr_context_create(0, &ctx) == HMR_OUT_OF_MEMORY);
    CHECK(ctx == nullptr);
    CHECK_ERROR("host allocation failed");
    CHECK(hmr_test_fail("a message that needs a host allocation to be stored") ==
          HMR_OUT_OF_MEMORY);
    CHECK_ERROR("host allocation failed");
    CHECK_STREAMS(0);
    fake::deny_new = false;
    CHECK(hmr_test_fail("a message stored once memory is back") == HMR_INVALID);
    CHECK_ERROR("a message stored once memory is back");
    create_and_destroy();
}

// Another exception, whose message can't be stored without memory.
void exception_without_memory() {
    const std::runtime_error error("the body's error");
    auto body = [&]() -> hmr_status { throw error; };
    CHECK(hmr::guarded(body) == HMR_INTERNAL_ERROR);
    CHECK_ERROR("internal error: the body's error");
    fake::deny_new = true;
    CHECK(hmr::guarded(body) == HMR_INTERNAL_ERROR);
    CHECK_ERROR("internal error (and its message couldn't be stored: host allocation failed)");
    fake::deny_new = false;
}

struct Mode {
    const char *name;
    void (*run)();
};

const Mode modes[] = {
    {"late-failure", late_failure},
    {"probe-failure", probe_failure},
    {"probe-then-destroy-failure", probe_then_destroy_failure},
    {"destroy-failure-without-memory", destroy_failure_without_memory},
    {"destroy-failure", destroy_failure},
    {"destroy-failure-without-memory-explicit", destroy_failure_without_memory_explicit},
    {"probe-failure-without-memory", probe_failure_without_memory},
    {"persistent-oom", persistent_oom},
    {"exception-without-memory", exception_without_memory},
};

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--list") == 0) {
        for (const Mode &m : modes) std::printf("%s\n", m.name);
        return 0;
    }
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s (--list | MODE)\n", argv[0]);
        return 2;
    }
    for (const Mode &m : modes) {
        if (std::strcmp(argv[1], m.name) == 0) {
            m.run();
            return failures == 0 ? 0 : 1;
        }
    }
    std::fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
