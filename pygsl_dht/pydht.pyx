cimport gsl_dht
cimport numpy as np
import numpy as np

cdef class DHT:
    """
    Wrapper per a calcular la transformada de Hankel utilitzant la llibreria GSL.
    Es tracta d'un objecte que crea un pla de transformació en inicialitzar-se amb
    la mida especificada. Després, accepta arrays de tipus double i en guarda la
    transformada a l'array result. Les posicions i freqüències es poden demanar a través dels
    mètodes get_x_sample i get_k_sample.
    """
    cdef gsl_dht.gsl_dht *_t
    cdef size_t size
    cdef double[:] result_view
    cdef np.ndarray result

    def __cinit__(self, size_t size):
        self.size = size

        # Inicio la transformació
        self._t = gsl_dht.gsl_dht_alloc(size)
        self.result = np.zeros(size)
        self.result_view = self.result

    cdef int _init(self, double nu, double xmax):
        return gsl_dht.gsl_dht_init(self._t, nu, xmax)

    def init(self, nu, xmax):
        """Inicia la transformació, d'ordre nu i extensió màxima xmax."""
        nu = float(nu)
        xmax = float(xmax)
        error = self._init(nu, xmax)
        if error:
            raise MemoryError("Could not create transform.")

    cdef void _get_x_samples(self, double[:] xsamples):
        cdef int i
        for i in range(self.size):
            xsamples[i] = gsl_dht.gsl_dht_x_sample(self._t, i)

    def get_x_samples(self):
        cdef np.ndarray[double, ndim=1, mode="c"] xsamples = np.zeros(self.size)
        cdef double[:] xsamples_view = xsamples
        self._get_x_samples(xsamples_view)
        return xsamples

    cdef void _get_k_samples(self, double[:] xsamples):
        cdef int i
        for i in range(self.size):
            xsamples[i] = gsl_dht.gsl_dht_k_sample(self._t, i)

    def get_k_samples(self):
        cdef np.ndarray[double, ndim=1, mode="c"] ksamples = np.zeros(self.size)
        cdef double[:] ksamples_view = ksamples
        self._get_k_samples(ksamples_view)
        return ksamples

    cdef void _apply(self, double[:] function):
        gsl_dht.gsl_dht_apply(self._t, &function[0], &self.result_view[0])

    def apply(self, function):
        if isinstance(function, np.ndarray):
            self._apply(function)
            return self.result

        else:
            raise TypeError("Expected 1D ndarray")

    def __dealloc__(self):
        gsl_dht.gsl_dht_free(self._t)
