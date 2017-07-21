CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)


IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing Torch-mklnn through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()

FIND_PACKAGE(Torch REQUIRED)

set(CMAKE_C_COMPILER icc)
IF (NOT WIN32)
SET(CMAKE_C_FLAGS "-std=c99 -Werror=implicit-function-declaration -qopenmp -O3 ${CMAKE_C_FLAGS}")
ENDIF (NOT WIN32)

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/mkltorch")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")
#MESSAGE("mkltorch path =  ${Torch_INSTALL_INCLUDE}/mkltorch, mkl path = $ENV{MKLML_LIBRARY}")

FILE(GLOB luasrc ./*.lua)
set(src  init.c)
MESSAGE(STATUS "src =" "${src}")
ADD_TORCH_PACKAGE(mklnn "${src}" "${luasrc}" "Intel MKLNN")

TARGET_LINK_LIBRARIES(mklnn mklml_intel mkltorch)
