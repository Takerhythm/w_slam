add_executable(run_kitti_stereo run_kitti_stereo.cpp)
target_link_libraries(run_kitti_stereo slam ${THIRD_PARTY_LIBS})

add_executable(run_euroc_stereo run_euroc_stereo.cpp)
target_link_libraries(run_euroc_stereo slam ${THIRD_PARTY_LIBS})

#add_executable(run_euroc_stereo_with_imu run_euroc_stereo_with_imu.cpp)
#target_link_libraries(run_euroc_stereo_with_imu slam ${THIRD_PARTY_LIBS})
