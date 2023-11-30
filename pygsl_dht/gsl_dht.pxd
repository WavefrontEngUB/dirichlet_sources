cdef extern from "gsl/gsl_dht.h":

    ctypedef struct gsl_dht:
        pass

    # Create a new transform object for a given size
    # sampling array on the domain [0, xmax].
    gsl_dht *gsl_dht_alloc(size_t size)
    gsl_dht *gsl_dth_new(size_t size, double nu, double xmax)

    # Recalculate a transform object for given values of nu, xmax.
    # You cannot change the size of the object since the internal
    # allocation is reused.
    int gsl_dht_init(gsl_dht *t, double nu, double xmax)

    # The n'th computed x sample point for a given transform.
    # 0 <= n <= size-1
    double gsl_dht_x_sample(const gsl_dht *t, int n)

    # The n'th computed k sample point for a given transform.
    # 0 <= n <= size-1
    double gsl_dht_k_sample(const gsl_dht *t, int n)

    # Free a transform object.
    void gsl_dht_free(gsl_dht *t)

    # Perform a transform on a sampled array.
    # f_in[0] ... f_in[size-1] and similarly for f_out[]
    int gsl_dht_apply(const gsl_dht *t, double *f_in, double *f_out)

