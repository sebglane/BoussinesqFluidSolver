##
#  CMake script
##

#
# Is the cmake version sufficient?
#
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

#
# Is the deal.ii library installed?
#
FIND_PACKAGE(deal.II 9.0.1 REQUIRED QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()


#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_UMFPACK) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_UMFPACK = ON
One or all of these are OFF in your installation but are required for this tutorial step."
    )
ENDIF()

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")


DEAL_II_INITIALIZE_CACHED_VARIABLES()
ADD_SUBDIRECTORY(source)
