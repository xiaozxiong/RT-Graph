set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")

find_path(OptiX_INCLUDE_DIR NAMES optix.h PATHS "${OptiX_INSTALL_DIR}/include" NO_DEFAULT_PATH)
find_path(OptiX_INCLUDE_DIR NAMES optix.h)

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

macro(cuda_compile_and_embed output_var cuda_file)
    set(c_var_name ${output_var})
    cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS --generate-line-info -use_fast_math --keep)
    list(GET ptx_files 0 ptx_file)
    set(embedded_file ${ptx_file}_embedded.c)
    #  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
    add_custom_command(
        OUTPUT ${embedded_file}
        COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
        DEPENDS ${ptx_file}
        COMMENT "compiling (and embedding ptx from) ${cuda_file}"
    )
    set(${output_var} ${embedded_file})
endmacro()

if(NOT OptiX_INCLUDE_DIR)
    OptiX_report_error("OptiX headers (optix.h and friends) not found." TRUE headers )
else()
    set(OptiX_FOUND true)
endif()

