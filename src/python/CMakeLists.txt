set(PYTHON_HEADERS
)

set(PYTHON_SOURCES
    bindings.cc
)

# build static libray
add_library(pyambit SHARED ${PYTHON_SOURCES} ${PYTHON_HEADERS})
add_dependencies(pyambit ambit)

if(ENABLE_CYCLOPS)
    find_library(CYCLOPS_LIB NAMES ctf PATHS ${CYCLOPS} "${CYCLOPS}/lib/")
    find_library(El_LIB El ${ELEMENTAL})
    find_library(PMRRR_LIB pmrrr "${ELEMENTAL}/external/pmrrr")
    target_link_libraries(pyambit ${CYCLOPS_LIB} ${El_LIB} ${PMRRR_LIB} ${MPI_LIBRARIES})
    set_target_properties(pyambit PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(pyambit PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

if(ENABLE_GA)
    execute_process(COMMAND ${GA}/bin/ga-config -flibs OUTPUT_VARIABLE GA_FLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GA}/bin/ga-config -ldflags OUTPUT_VARIABLE GA_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${GA}/bin/ga-config -libs OUTPUT_VARIABLE GA_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    target_link_libraries(pyambit ${GA_FLIBS} ${GA_LDFLAGS} ${GA_LIBS} ${MPI_LIBRARIES})
endif()

target_link_libraries(pyambit ambit ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# Ensure the Python module is appropriately named
set_target_properties(pyambit PROPERTIES PREFIX "")
set_target_properties(pyambit PROPERTIES SUFFIX ${PYTHON_MODULE_EXTENSION})

install(TARGETS pyambit DESTINATION lib/ambit)

