cmake_minimum_required(VERSION 3.10)
project(pai_policy_inference)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(pai_tensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(pai_policy_inference_core SHARED src/signature.cpp
                                             src/tensor_view.cpp)
target_include_directories(
  pai_policy_inference_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(
  pai_policy_inference_core PUBLIC yaml-cpp::yaml-cpp rclcpp::rclcpp
                                   ${pai_tensor_msgs_TARGETS})

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS pai_policy_inference_core
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  set(FIXTURE_DIR_PATH ${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures)

  ament_add_gtest(test_signature test/test_signature.cpp)
  target_link_libraries(test_signature pai_policy_inference_core)
  target_compile_definitions(
    test_signature PRIVATE FIXTURE_DIR="${FIXTURE_DIR_PATH}")

  ament_add_gtest(test_tensor_view test/test_tensor_view.cpp)
  target_link_libraries(test_tensor_view pai_policy_inference_core)
  target_compile_definitions(
    test_tensor_view PRIVATE FIXTURE_DIR="${FIXTURE_DIR_PATH}")
endif()

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rclcpp_components pai_tensor_msgs std_msgs
                          yaml-cpp)
ament_package()
