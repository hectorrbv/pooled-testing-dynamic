"""Comprobaciones pequeñas para la sesión de estudio; no modifica resultados previos.

Ejecutar desde cualquier directorio con python3 y redirigir stdout si se desea.
No sustituye la validación independiente pathwise pendiente en el plan.
"""
import json
import sys
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from augmented.bm17_toy_solver import SolverLaminar
from augmented.bellman_tipos import crear_solver
from augmented.densidad_companion import PoliticasDensidad
from augmented.indice_lagrangiano import PoliticaLagrangiana


def par(n, q, conv):
    s = SolverLaminar(dict.fromkeys(range(n), 1-q),
                      dict.fromkeys(range(n), F(1)), 2, conv)
    return {
        'par_primero_continuacion_optima': float(s.valor_forzando_primera(
            frozenset(range(n)), (), 2, ('open', (0, 1)))),
        'optimo': float(s.V(frozenset(range(n)), (), 2)),
    }


q = F(3, 10)
pz = q*q*(2+q) + 2*q*(1-q) + (1-q)**2*q
strict = q*q*(2+q) + 2*q*(1-q)*F(1, 2) + (1-q)**2*q
out = {'fecha': '2026-09-21', 'alcance': 'Comprobaciones de estudio, no suite completa; score inmediato corregido'}
out['ejemplo_b2'] = {
    'q_sano': float(q), 'B': 2, 'G': 2, 'n': 4,
    'pz_a_mano': str(pz), 'strict_par_a_mano': str(strict),
    'singleton': str(2*q),
    'posterior_zero': par(4, q, 'posterior_zero'),
    'strict': par(4, q, 'strict'),
}
assert pz == F(387, 500) and strict == F(141, 250)
out['formula_del_plan_fuera_del_regimen'] = {
    'n': 4, 'q_sano': .9,
    'posterior_zero': par(4, F(9, 10), 'posterior_zero'),
    'strict': par(4, F(9, 10), 'strict'),
    'pq': .09,
}
out['compresion_por_tipos'] = []
for n in (12, 120, 500):
    v = crear_solver(.7, 3)
    val = v(n, (), 3)
    out['compresion_por_tipos'].append(
        {'n': n, 'B': 3, 'G': 3, 'q_sano': .3,
         'valor': val, 'estados': v.cache_info().currsize})

p = {0:F(9,10), 1:F(33,40), 2:F(7,8), 3:F(4,5), 4:F(19,20), 5:F(17,20)}
u = {0:2, 1:1, 2:1, 3:1, 4:4, 5:2}
s = SolverLaminar(p, u, 4, 'posterior_zero')
opt = s.V(frozenset(p), (), 3)
pol = PoliticaLagrangiana(p, u, 4, lam=.001, horizonte=3)
w = pol.valor(frozenset(p), (), 3)
out['contraejemplo_y_pi_L'] = {
    'B':3, 'G':4, 'convencion':'posterior_zero', 'clase':'pathwise',
    'optimo_fraccion':str(opt), 'optimo':float(opt),
    'primera_accion_optima':s.politica(frozenset(p), (), 3),
    'lambda':.001, 'horizonte':3, 'no_paralisis':True,
    'pi_L_valor':w, 'pi_L_ratio':w/float(opt),
}

p3 = dict.fromkeys(range(3), F(1, 2))
u3 = dict.fromkeys(range(3), 1)
act = ('ref', ((0, 1, 2), 2), (0,))
d = PoliticasDensidad(p3, u3, 3)
s3 = SolverLaminar(p3, u3, 3, 'posterior_zero')
out['regresion_score_inmediato'] = {
    'estado':'atomo de 3 personas, 2 infectados, u=1, prior homogeneo',
    'accion':'probar una persona',
    'score_del_codigo': d._score_inmediato(act),
    'cobro_correcto_a_mano': str(F(1, 3)),
    'cobro_bellman_un_paso':float(s3.valor_forzando_primera(
        frozenset(), (((0,1,2),2),), 1, act)),
    'score_antes_de_corregir':str(F(5, 3)),
    'estado_revision':'Corregido el 21-sep; cotejado por enumeración en tests_bellman_perfiles.py',
}
assert abs(d._score_inmediato(act) - 1/3) < 1e-12
out['ancla_cotas_realizables'] = {
    'q_sano':.05, 'G':16, 'B':7,
    'k_posterior_zero':3, 'cota_posterior_zero':1-.95**48,
    'k_reserva_estricta_del_plan':2, 'cota_reserva_estricta':1-.95**32,
    'singleton':7*.05,
    'nota':'Son cotas inferiores de construcciones, no valores optimos certificados.',
}
print(json.dumps(out, indent=2, ensure_ascii=False))
