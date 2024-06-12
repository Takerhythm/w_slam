file(GLOB_RECURSE src_files *.cpp *.cc)
#message(STATUS "Source files: ${src_files}")
add_library(slam SHARED
        ${src_files}
#        frame.cpp
#        mappoint.cpp
#        map.cpp
#        camera.cpp
#        config.cpp
#        feature.cpp
#        frontend.cpp
#        backend.cpp
#        viewer.cpp
#        System.cpp
#        dataset.cpp
)

target_link_libraries(slam ${THIRD_PARTY_LIBS})
