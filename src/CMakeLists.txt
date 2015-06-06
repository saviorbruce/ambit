set(TENSOR_HEADERS
    ${PROJECT_SOURCE_DIR}/include/ambit/tensor.h
    ${PROJECT_SOURCE_DIR}/include/ambit/print.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/file.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/io.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/iwl.h
    ${PROJECT_SOURCE_DIR}/include/ambit/io/manager.h
    ${PROJECT_SOURCE_DIR}/include/ambit/helpers/psi4/io.h
    math/math.h
    tensor/core/core.h
    tensor/disk/disk.h
    tensor/indices.h
    tensor/globals.h
    tensor/macros.h
    tensor/tensorimpl.h
    ${PROJECT_SOURCE_DIR}/include/ambit/timer.h
)

set(TENSOR_SOURCES
    helpers/psi4/io.cc
    io/file.cc
    io/iwl.cc
    io/manager.cc
    math/blas.cc
    math/lapack.cc
    tensor/core/core.cc
    tensor/disk/disk.cc
    tensor/indices.cc
    tensor/globals.cc
    tensor/labeled_tensor.cc
    tensor/print.cc
    tensor/slice.cc
    tensor/sliced_tensor.cc
    tensor/tensor.cc
    tensor/tensorimpl.cc
    tensor/timer.cc
    blocked_tensor/blocked_tensor.cc
)

# if we have MPI and Cyclops is enabled
if(ENABLE_CYCLOPS)
    list(APPEND TENSOR_HEADERS tensor/cyclops/cyclops.h)
    list(APPEND TENSOR_SOURCES tensor/cyclops/cyclops.cc)
endif()
# if we have MPI and Global Array is enabled
if(ENABLE_GA)
    list(APPEND TENSOR_HEADERS tensor/global_array/global_array.h)
    list(APPEND TENSOR_SOURCES tensor/global_array/global_array.cc)
endif()

# build libray
add_library(ambit SHARED ${TENSOR_SOURCES} ${TENSOR_HEADERS})

if(ENABLE_CYCLOPS)
    find_library(CYCLOPS_LIB NAMES ctf PATHS ${CYCLOPS} "${CYCLOPS}/lib/")
    find_library(El_LIB El ${ELEMENTAL})
    find_library(PMRRR_LIB pmrrr "${ELEMENTAL}/external/pmrrr")
    target_link_libraries(ambit ${CYCLOPS_LIB} ${El_LIB} ${PMRRR_LIB} ${MPI_LIBRARIES})
endif()

if(ENABLE_GA)
    execute_process(COMMAND ${GA}/bin/ga-config -flibs OUTPUT_VARIABLE GA_FLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GA}/bin/ga-config -ldflags OUTPUT_VARIABLE GA_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GA}/bin/ga-config -libs OUTPUT_VARIABLE GA_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_link_libraries(ambit ${GA_FLIBS} ${GA_LDFLAGS} ${GA_LIBS} ${MPI_LIBRARIES})
endif()

target_link_libraries(ambit ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${EXTERNAL_LIBS})

install(TARGETS ambit LIBRARY DESTINATION lib)

if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

