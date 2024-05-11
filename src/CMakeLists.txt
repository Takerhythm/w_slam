add_library(slam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        config.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        System.cpp
#        dataset.cpp
)

target_link_libraries(slam
        ${THIRD_PARTY_LIBS})
