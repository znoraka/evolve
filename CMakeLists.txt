project( evolve )

cmake_minimum_required(VERSION 2.6)

ADD_DEFINITIONS(
  -std=c++11
  -w
  -lpthread
  -Ofast
  # -g
  )

find_package( OpenCV )

set(EXECUTABLE_OUTPUT_PATH bin/${CMAKE_BUILD_TYPE})

file(

  GLOB_RECURSE

  source_files

  src/*

  )

add_executable(

  evolve

  ${source_files}

  )

target_link_libraries(evolve ${OpenCV_LIBS} -lpthread -lX11)
