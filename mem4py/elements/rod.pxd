# cython: profile=False, cdivision=True, boundcheck=False, wraparound=False, nonecheck=False, language_level=3
cdef int rodBendingFintAndK(double [:] X,
                            double [:] Y,
                            double [:] Z,
                            int [:, ::1] NBend,
                            Py_ssize_t el,
                            double kB,
                            unsigned int dim,
                            unsigned int index,
                            double [:] Fint,
                            double [:] data,
                            double [:] diagK,
                            unsigned int [:] order,
                            double * SE) nogil
