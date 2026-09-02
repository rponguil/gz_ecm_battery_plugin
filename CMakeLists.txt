cmake_minimum_required(VERSION 3.16)
project(gz_ecm_battery_plugin)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Same Gazebo Sim version pinning approach as as2_gazebo_assets (verified
# working against Gazebo Harmonic in this workspace) -- keeps this plugin
# buildable in the same environment without version drift surprises.
set(GZ_SIM "gz-sim8")
set(GZ_COMMON "gz-common5")
set(GZ_TRANSPORT "gz-transport13")
set(GZ_MSGS "gz-msgs10")
set(GZ_PLUGIN "gz-plugin2")

find_package(${GZ_SIM} REQUIRED)
find_package(${GZ_COMMON} REQUIRED)
find_package(${GZ_TRANSPORT} REQUIRED)
find_package(${GZ_MSGS} REQUIRED)
find_package(${GZ_PLUGIN} REQUIRED COMPONENTS register)

add_library(gz_ecm_battery_plugin SHARED
  src/EscBatteryPlugin.cc
)

target_include_directories(gz_ecm_battery_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(gz_ecm_battery_plugin PUBLIC
  ${GZ_SIM}
  ${GZ_COMMON}
  ${GZ_TRANSPORT}
  ${GZ_MSGS}
)

install(
  TARGETS gz_ecm_battery_plugin
  DESTINATION lib)

install(
  DIRECTORY include/
  DESTINATION include)

install(
  DIRECTORY worlds/
  DESTINATION share/${PROJECT_NAME}/worlds)

option(BUILD_TESTING "Build gz_ecm_battery_plugin tests" ON)
if(BUILD_TESTING)
  find_package(GTest QUIET)
  if(GTest_FOUND)
    enable_testing()
    add_executable(test_esc_model test/test_esc_model.cc)
    target_include_directories(test_esc_model PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(test_esc_model GTest::gtest GTest::gtest_main)
    add_test(NAME test_esc_model COMMAND test_esc_model)
  else()
    message(STATUS "gz_ecm_battery_plugin: GTest not found, skipping tests "
                    "(apt install libgtest-dev, or run inside a ROS 2 "
                    "workspace which provides it via ros-jazzy-gtest-vendor)")
  endif()
endif()

# Support both plain `cmake --build` and `colcon build` (ament_cmake) without
# forcing a ROS 2 dependency on the plugin code itself.
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
  ament_package()
endif()
