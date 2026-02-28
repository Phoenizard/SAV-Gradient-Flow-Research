"""Analyze Phase 1 results for report."""
import torch
import sys
sys.path.insert(0, '.')

def analyze(results, title):
    print(f"\n=== {title} ===")
    for r in results:
        e = r.get('energy', [])
        finite_e = [x for x in e if x == x and x != float('inf')]
        if len(finite_e) > 1:
            violations = sum(1 for i in range(len(finite_e)-1) if finite_e[i+1] > finite_e[i] * 1.001)
        else:
            violations = 0
        rv = r.get('r_values', [])
        if rv:
            mod_e = [rv[i]**2 for i in range(len(rv))]
            mod_violations = sum(1 for i in range(len(mod_e)-1) if mod_e[i+1] > mod_e[i] * 1.001)
            print(f"  {r['method']:15s}: train={r['final_train_loss']:.4e}, test={r['final_test_loss']:.4e}, "
                  f"time={r['wall_time']:.1f}s")
            print(f"    r: {rv[0]:.4e} -> {rv[-1]:.4e}")
            print(f"    energy violations (true): {violations}/{max(len(finite_e)-1,1)}")
            print(f"    energy violations (mod r^2): {mod_violations}/{max(len(mod_e)-1,1)}")
        else:
            print(f"  {r['method']:15s}: train={r['final_train_loss']:.4e}, test={r['final_test_loss']:.4e}, "
                  f"time={r['wall_time']:.1f}s")
            print(f"    energy violations: {violations}/{max(len(finite_e)-1,1)}")

# Fig 1
fig1 = torch.load('results/phase1_sav/fig1_all_results.pt', weights_only=False)
analyze(fig1, "FIG 1 (Example 1, D=20, m=1000, lr=0.6, lambda=10, 8000 epochs)")

# Fig 7
fig7 = torch.load('results/phase1_sav/fig7_all_results.pt', weights_only=False)
analyze(fig7, "FIG 7 (Example 2, D=40, m=100, lr=0.01, lambda=0, 10000 epochs)")

# Fig 2 (old run)
fig2 = torch.load('results/phase1_sav/fig2_all_results.pt', weights_only=False)
analyze(fig2, "FIG 2 OLD (Example 1, D=40, m=1000, lr=0.2, lambda=0, 10000 epochs)")

# Paper comparison summary
print("\n=== PAPER COMPARISON (WORKFLOW.md Step 7) ===")
fig7_methods = {r['method']: r for r in fig7}
sgd = fig7_methods.get('SGD', {})
van = fig7_methods.get('Vanilla SAV', {})
res = fig7_methods.get('Restart SAV', {})
rel = fig7_methods.get('Relax SAV', {})

sgd_test = sgd.get('final_test_loss', float('inf'))
van_test = van.get('final_test_loss', float('inf'))
res_test = res.get('final_test_loss', float('inf'))
rel_test = rel.get('final_test_loss', float('inf'))

print(f"  SGD test: {sgd_test:.4e}")
print(f"  Vanilla test: {van_test:.4e}")
print(f"  Restart test: {res_test:.4e}")
print(f"  Relax test: {rel_test:.4e}")
print()
print(f"  1. SAV faster than SGD? Restart={res_test < sgd_test} (Restart {res_test:.4e} vs SGD {sgd_test:.4e})")
print(f"  2. Restart <= Vanilla? {res_test <= van_test}")
print(f"  3. Relax <= Restart? {rel_test <= res_test}")
print(f"  4. Vanilla energy non-increasing (mod r^2)? Check above")
print(f"  5. Fig 1: GD diverges, SAV stable? Check above")
