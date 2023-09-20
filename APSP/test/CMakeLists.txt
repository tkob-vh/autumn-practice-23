include("${PROJECT_SOURCE_DIR}/etc/utils.cmake")

foreach(SubDir ${SubDirs})
    add_executable(apsp_test_${SubDir} apsp_test.cc)
    target_link_libraries(apsp_test_${SubDir} PRIVATE graph apsp_${SubDir})
endforeach()
