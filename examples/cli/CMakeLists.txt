add_executable(voicedetect-cli main.cpp)
target_link_libraries(voicedetect-cli PRIVATE voicedetect)
target_include_directories(voicedetect-cli PRIVATE ${CMAKE_SOURCE_DIR}/src)
