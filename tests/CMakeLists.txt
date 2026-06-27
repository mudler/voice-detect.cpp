function(vd_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE voicedetect)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

vd_add_test(test_smoke)
vd_add_test(test_fbank)
vd_add_test(test_encoder)
vd_add_test(test_embed)
vd_add_test(test_verify)
vd_add_test(test_analyze)
vd_add_test(test_age_gender)
vd_add_test(test_blocked)
vd_add_test(test_cache_safety)

# Manual benchmark target (not registered as a ctest; run directly).
add_executable(bench_wespeaker bench_wespeaker.cpp)
target_link_libraries(bench_wespeaker PRIVATE voicedetect)
target_include_directories(bench_wespeaker PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)

add_executable(bench_eres2net bench_eres2net.cpp)
target_link_libraries(bench_eres2net PRIVATE voicedetect)
target_include_directories(bench_eres2net PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/third_party)

# Model/baseline-dependent tests read fixtures/baselines via paths relative to
# the project root, and skip (exit 77) when the baseline env var is unset.
set_tests_properties(test_fbank test_encoder test_embed test_verify test_analyze test_age_gender test_cache_safety PROPERTIES
                     LABELS "model"
                     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Python converter check (skips with exit 77 when the venv/model are absent).
find_program(VOICEDETECT_PY NAMES python3 HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
if(VOICEDETECT_PY)
  add_test(NAME check_convert COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py)
  set_tests_properties(check_convert PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  # Analyze (wav2vec2 emotion) converter check: a SEPARATE analyze GGUF, skips
  # independently of the speaker-encoder check when its HF model is absent.
  add_test(NAME check_convert_analyze COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py --analyze)
  set_tests_properties(check_convert_analyze PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  add_test(NAME check_baseline_analyze COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py --analyze)
  set_tests_properties(check_baseline_analyze PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  # Age/gender (audeering wav2vec2-large-robust) converter + baseline checks: a
  # SEPARATE analyze GGUF, skips independently when its HF model is absent.
  add_test(NAME check_convert_age_gender COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_convert.py --age-gender)
  set_tests_properties(check_convert_age_gender PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  add_test(NAME check_baseline_age_gender COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py --age-gender)
  set_tests_properties(check_baseline_age_gender PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  add_test(NAME check_manifest COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_manifest.py)
  set_tests_properties(check_manifest PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
  add_test(NAME check_baseline COMMAND ${VOICEDETECT_PY} ${CMAKE_SOURCE_DIR}/tests/python/check_baseline.py)
  set_tests_properties(check_baseline PROPERTIES SKIP_RETURN_CODE 77 LABELS "model")
endif()
