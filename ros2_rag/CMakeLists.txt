cmake_minimum_required(VERSION 3.8)
project(ros2_rag)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_compile_options(-std=c++17)

# find dependencies
find_package(ament_cmake REQUIRED)

## Testing
if(BUILD_TESTING)
  find_package(launch_testing_ament_cmake REQUIRED)
  find_package(ros_testing REQUIRED)

  add_launch_test(test/test_lifecycle_launch.py TIMEOUT "180")

endif()

ament_package()
