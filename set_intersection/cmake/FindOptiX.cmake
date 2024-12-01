
# set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")
# set(CACHE VAR “value” [FORCE]): set caceh variable
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")

# The distribution contains only 64 bit libraries.  Error when we have been mis-configured.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(WIN32)
        message(SEND_ERROR "Make sure when selecting the generator, you select one with Win64 or x64.")
    endif()
        message(FATAL_ERROR "OptiX only supports builds configured for 64 bits.")
endif()

# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(NOT APPLE)
    set(bit_dest "64")
else()
    set(bit_dest "")
endif()


# finid_path: find a directory containing the named file, <VAR> is created to store the result of this command.
# If nothing is found, the result will be <VAR>-NOTFOUND.
# NAME: Specify one or more possible names for the file in a directory.
# PATHS: Specify directories to search in addition to the default locations.
# Include
find_path(OptiX_INCLUDE_DIRS
    NAMES optix.h
    PATHS "${OptiX_INSTALL_DIR}/include"
    NO_DEFAULT_PATH
)
find_path(OptiX_INCLUDE_DIRS
    NAMES optix.h
)

# Check to make sure we find the OptiX
function(OptiX_report_error error_message required component )
    if(DEFINED OptiX_FIND_REQUIRED_${component} AND NOT OptiX_FIND_REQUIRED_${component})
        set(required FALSE)
    endif()
    if(OptiX_FIND_REQUIRED AND required)
        message(FATAL_ERROR "${error_message}  Please locate before proceeding.")
    else()
        if(NOT OptiX_FIND_QUIETLY)
          message(STATUS "${error_message}")
        endif(NOT OptiX_FIND_QUIETLY)
    endif()
endfunction()

if(NOT OptiX_INCLUDE_DIRS)
    OptiX_report_error("OptiX headers (optix.h and friends) not found." TRUE headers)
else()
    set(OptiX_FOUND true)
endif()

