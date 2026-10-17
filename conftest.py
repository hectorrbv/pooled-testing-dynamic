"""Configuracion de pytest para todo el repo.

`augmented.core.test_result(pool_mask, z_mask)` es una funcion de PRODUCCION
--- devuelve el conteo que arroja un pool bajo un perfil latente --- pero su
nombre empieza con `test_`, asi que pytest la colecciona como si fuera una
prueba en cada modulo que la importa y falla pidiendo un fixture `pool_mask`
que no existe.

La marca `__test__ = False` es el mecanismo estandar de pytest para decir
"esto no es una prueba".  Se pone aqui, y no en `core.py`, para no tocar el
modulo de produccion: no cambia el comportamiento de la funcion en absoluto.
"""

from augmented import core

core.test_result.__test__ = False
