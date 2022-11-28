cmake_minimum_required( VERSION 3.8 )
set( VCPKG_BASE_DIR "C:/Users/optocloud/source/repos/vcpkg" )
set( CMAKE_TOOLCHAIN_FILE "${VCPKG_BASE_DIR}/scripts/buildsystems/vcpkg.cmake" )
message (STATUS ${VCPKG_BASE_DIR})
message (STATUS ${CMAKE_TOOLCHAIN_FILE})

set( project main )
project( ${project}
    DESCRIPTION "Testing"
    LANGUAGES CXX
)

if( NOT CMAKE_CL_64 )
    set( OpenCV_DIR "${VCPKG_BASE_DIR}/installed/x86-windows/share/opencv4" )
else()
    set( OpenCV_DIR "${VCPKG_BASE_DIR}/installed/x64-windows/share/opencv4" )
endif()

message (STATUS ${OpenCV_DIR})

set( CMAKE_CXX_STANDARD 20 )
set( THREADS_PREFER_PTHREAD_FLAG ON )
find_package( Threads REQUIRED )

find_package( OpenCV REQUIRED )

# If release, set max optimization
if( CMAKE_BUILD_TYPE STREQUAL "Release" )
	set( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2" )
endif()

set( ${project}_src
	main.cpp
)
set( ${project}_headers
)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( ${project}
    ${${project}_src}
    ${${project}_headers}
)

target_link_libraries( main 
    Threads::Threads
    ${OpenCV_LIBRARIES}
)