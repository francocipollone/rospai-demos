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

# ONNX Runtime comes from the conda prefix (conda-forge onnxruntime-cpp), which
# is on CMAKE_PREFIX_PATH; it ships no CMake config, so wrap it ourselves.
find_path(
  ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATH_SUFFIXES onnxruntime/core/session onnxruntime REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime REQUIRED)
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(
  onnxruntime::onnxruntime
  PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
             INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

add_library(
  pai_policy_inference_core SHARED src/signature.cpp src/tensor_view.cpp
                                   src/ort_backend.cpp src/inference_core.cpp)
target_include_directories(
  pai_policy_inference_core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include/${PROJECT_NAME}>)
target_link_libraries(
  pai_policy_inference_core PUBLIC yaml-cpp::yaml-cpp rclcpp::rclcpp
                                   ${pai_tensor_msgs_TARGETS})
# PRIVATE: ONNX Runtime headers never appear in the installed headers (pimpl),
# so downstream packages need no onnxruntime target; the shared library
# resolves libonnxruntime.so at load time from the environment.
target_link_libraries(
  pai_policy_inference_core PRIVATE onnxruntime::onnxruntime)

add_library(pai_policy_inference_node SHARED src/inference_node.cpp)
target_link_libraries(
  pai_policy_inference_node
  PUBLIC pai_policy_inference_core rclcpp::rclcpp
         rclcpp_components::component ${pai_tensor_msgs_TARGETS})
rclcpp_components_register_node(
  pai_policy_inference_node PLUGIN "pai_policy_inference::InferenceNode"
  EXECUTABLE inference_node_component)

add_executable(inference_node src/inference_node_main.cpp)
target_link_libraries(inference_node pai_policy_inference_node)

install(DIRECTORY include/ DESTINATION include/${PROJECT_NAME})
install(
  TARGETS pai_policy_inference_core pai_policy_inference_node
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

  ament_add_gtest(test_inference_core test/test_inference_core.cpp)
  target_link_libraries(test_inference_core pai_policy_inference_core)
  target_compile_definitions(
    test_inference_core PRIVATE FIXTURE_DIR="${FIXTURE_DIR_PATH}")

  ament_add_gtest(test_tensor_view test/test_tensor_view.cpp)
  target_link_libraries(test_tensor_view pai_policy_inference_core)
  target_compile_definitions(
    test_tensor_view PRIVATE FIXTURE_DIR="${FIXTURE_DIR_PATH}")
endif()

install(TARGETS inference_node DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(rclcpp rclcpp_components pai_tensor_msgs std_msgs
                          yaml-cpp)
ament_package()
