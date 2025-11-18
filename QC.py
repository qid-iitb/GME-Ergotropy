import numpy as np
import scipy as sp
import pickle

import os, json, math, argparse, warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import datetime
from scipy.optimize import OptimizeResult, minimize
from itertools import combinations

from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.transpiler import CouplingMap
from rustworkx.visualization import graphviz_draw

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler, QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator

from qiskit.quantum_info import partial_trace
from qiskit.quantum_info import Statevector

from qiskit.synthesis import SuzukiTrotter
from qiskit_addon_utils.problem_generators import generate_time_evolution_circuit

from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit

import quimb.tensor
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator

from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit_addon_aqc_tensor.simulation import compute_overlap

from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity as StF
#from qiskit_addon_aqc_tensor.objective import OneMinusFidelity as StF

# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import efficient_su2 as EfficientSU2, Initialize

# For AER simulator
from qiskit_aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeMumbaiV2, FakePerth
from qiskit_aer.noise import NoiseModel

from qiskit_algorithms.optimizers import SPSA

# For Algorithms
#from qiskit_algorithms.utils import algorithm_globals

def get_hamiltonian(L, g, h):
    XX_tuples = [("XX", [i, i + 1], -g) for i in range(0, L - 1)] + [("XX", [L-1, 0], -g)]
    Z_tuples = [("Z", [i], -h) for i in range(0, L)]
    hamiltonian = SparsePauliOp.from_sparse_list([*XX_tuples, *Z_tuples], num_qubits=L)
    return hamiltonian.simplify()

def get_battery_hamiltonian(L, party):
    Z_tuples = [("Z", [i], 1) for i in party]
    hamiltonian = SparsePauliOp.from_sparse_list([*Z_tuples], num_qubits=L)
    return hamiltonian.simplify()

def ansatz_circ(in_st,Nqubits,party,reps=2):
    entangler_map = [[party[i],party[i+1]] for i in range(len(party)-1)]
    if len(party)>1:
        ansatz = EfficientSU2(num_qubits=Nqubits,entanglement=entangler_map,
                              reps=reps,skip_unentangled_qubits=True,insert_barriers=False,initial_state=in_st,
                             flatten=True)
    else:
        a1 = EfficientSU2(num_qubits=1,
                              reps=reps,insert_barriers=False,
                             flatten=True)
        ansatz = in_st.compose(a1,qubits=party)

    return (ansatz,ansatz.num_parameters)

def time_evolve(L,g,h,aqc_evolution_time,aqc_max_iterations = 5000,aqc_ansatz_num_trotter_steps = 1):
    hamiltonian = get_hamiltonian(L,g,h)
    initial_state = QuantumCircuit(L)
    aqc_target_num_trotter_steps= int(aqc_evolution_time*200)

    aqc_target_circuit = initial_state.copy()
    aqc_target_circuit.compose(
        generate_time_evolution_circuit(
            hamiltonian,
            synthesis=SuzukiTrotter(reps=aqc_target_num_trotter_steps),
            time=aqc_evolution_time,
        ),
        inplace=True,
    )

    aqc_good_circuit = initial_state.copy()
    aqc_good_circuit.compose(
        generate_time_evolution_circuit(
            hamiltonian,
            synthesis=SuzukiTrotter(reps=aqc_ansatz_num_trotter_steps),
            time=aqc_evolution_time,
        ),
        inplace=True,
    )

    aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(aqc_good_circuit)

    simulator_settings = QuimbSimulator(quimb.tensor.CircuitMPS, autodiff_backend="jax")
    aqc_target_mps = tensornetwork_from_circuit(aqc_target_circuit, simulator_settings)

    good_mps = tensornetwork_from_circuit(aqc_good_circuit, simulator_settings)
    starting_fidelity = abs(compute_overlap(good_mps, aqc_target_mps)) ** 2
    print("Starting fidelity:", starting_fidelity)
        
    objective = StF(aqc_target_mps, aqc_ansatz, simulator_settings)
        
    result = minimize(
        objective,
        aqc_initial_parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": aqc_max_iterations},
        #callback=callback,
    )
    if result.status not in (
        0,
        1,
        99,
    ):  # 0 => success; 1 => max iterations reached; 99 => early termination via StopIteration
        #raise RuntimeError(
        print(
            f"Optimization failed: {result.message} (status={result.status})"
        )
    aqc_final_parameters = result.x
    aqc_final_circuit = aqc_ansatz.assign_parameters(aqc_final_parameters)

    return aqc_final_circuit

def cost_func_vqe(parameters, ansatz, hamiltonian, estimator):
    estimator_job = estimator.run([(ansatz, hamiltonian, [parameters])])
    estimator_result = estimator_job.result()[0]

    cost = estimator_result.data.evs[0]
    return cost

def transpile(H, ansatz, backend_answer = AerSimulator(), optimization_level_answer = 3):
    pm = generate_preset_pass_manager(backend=backend_answer,optimization_level=optimization_level_answer)
    isa_circuit = pm.run(ansatz)
    hamiltonian_isa = H.apply_layout(layout=isa_circuit.layout)
    return (hamiltonian_isa, isa_circuit)

def var_ergo_noiseless(L,evolution_qc,party,nrep=1,optimizer="BFGS"):
    evolution_qc.barrier()

    estimator = StatevectorEstimator()
    ansatz, nparam = ansatz_circ(evolution_qc,L,party,reps=nrep)
    x0 = np.ones(nparam)
    HB = get_battery_hamiltonian(L,party)
    result = minimize(cost_func_vqe, x0, args=(ansatz, HB, estimator), method=optimizer,tol=10E-6)#,options={'maxiter': iterations})
    return result

#========================================================================================================================================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def build_grid(t_vals, Nrep_vals, party, out_root, cfg):
    for t in t_vals:
        for Nrep in Nrep_vals:
            for p in party:
                yield (float(t), int(Nrep), p, cfg)

def run_one(task):
    t, nrep, p, cfg = task
    L, g, h = cfg["L"], cfg["g"], cfg["h"]
    out_root = cfg["out_root"]
    resume = bool(cfg.get("resume", True))

    run_dir = os.path.join(out_root, f"N={int(L):02d}",f"t={t:.3g}", f"nrep={nrep:.3g}", f"p={p:.3g}")
    
    ensure_dir(run_dir)
    
    # Resume: skip if summary.json exists
    summary_path = os.path.join(run_dir, "summary.json")
    if resume and os.path.exists(summary_path):
        return {"skip": True, "Nrep":nrep, "t":t, "party":p, "L":int(L), "g":g, "h":h}

    evol_st = time_evolve(L,g,h,t,aqc_ansatz_num_trotter_steps = 20)
    party = list(range(0,p))
    result = var_ergo_noiseless(L,evol_st,party,nrep=nrep,optimizer="L-BFGS-B")

    result_path = os.path.join(run_dir, "optimization_result.pkl")
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    # save summary
    summary = dict(
        L=L, g=g, h=h, t=t, nrep=nrep, party=party,
        E_passive = result.fun,
        nit=result.nit, nfev=result.nfev,
        success=result.success, message=result.message,
    )
    save_json(os.path.join(run_dir, "summary.json"), summary)

    return {
        "ok": True,
        "L": L, "g": g, "h": h, "Nrep": nrep, "t": t, "party": p,
        "E_ps": result.fun,
    }

def main():
    ap = argparse.ArgumentParser(description="Ergotropy QC.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cores", type=int, default=max(cpu_count()-1,1))
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--g", type=float, default=2.0)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--Nrepmin", type=int, default=4)
    ap.add_argument("--Nrepmax", type=int, default=8)
    ap.add_argument("--Nrepstep", type=int, default=1)
    ap.add_argument("--tmin", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=1.5)
    ap.add_argument("--tstep", type=float, default=0.25)
    # Resume toggle:
    ap.add_argument("--resume", choices=["yes","no"], default="yes", help="Skip runs with existing summary.json")
    args = ap.parse_args()
    
    out_root = os.path.abspath(args.out)
    ensure_dir(out_root)

    cfg = dict(
        out_root=out_root,
        L=args.L, g=args.g, h=args.h,
        resume=(args.resume == "yes"),
    )

    Nrep_vals  = list(range(args.Nrepmin, args.Nrepmax+1, args.Nrepstep))
    t_vals  = np.round(np.arange(args.tmin, args.tmax + 1e-12, args.tstep), 3)
    party = list(range(1,args.L))

    grid = list(build_grid(t_vals, Nrep_vals, party, out_root, cfg))
    print(f"[INFO] Starting pool with {args.cores} workers over {len(grid)} combos.")

    with Pool(processes=args.cores) as pool:
        it = pool.imap_unordered(run_one, grid, chunksize=1)
        with tqdm(total=len(grid), ncols=100, dynamic_ncols=True, leave=True) as pbar:
            for info in it:
                pbar.update(1)
                if info and info.get("ok", False):
                    tqdm.write(
                        f"[OK] L={info['L']} g={info['g']:.3g} h={info['h']:.3f} "
                        f"Nrep={info['Nrep']:02d}  t≈{info['t']:.3g}  "
                        f"Party≈{info['party']:02d} "
                        f"Epassive≈{info['E_ps']:.3g}  "
                    )
                elif info.get("skip"):
                    tqdm.write(f"[SKIP] L={info['L']} g={info['g']:.3g} h={info['h']:.3f} Nrep={info['Nrep']:02d} t≈{info['t']:.3g} Party≈{info['party']:02d} (resume)")
                else:
                    tqdm.write("[ERR] A run failed or returned no info.")
    print("[DONE] All runs finished.")

if __name__ == "__main__":
    main()

#========================================================================================================================================
'''Instructions to run:
python QC.py   --out ./QC_out_prerun   --cores 20 \
 --L 6 --g 2.0 \
 --h 1.0 \
 --Nrepmin 4 --Nrepmax 8 --Nrepstep 2 \
 --tmin 0.0 --tmax 1.5 --tstep 0.25 \
 --resume yes'''
