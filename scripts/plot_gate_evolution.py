import json
import matplotlib.pyplot as plt

with open('experiments/test_regime_attention_weekly/collapse_monitoring/collapse_monitor_latest.json') as f:
    history = json.load(f)

gates = history['regime_attention_gate_values']
# gates[epoch][regime][head]

epochs = range(len(gates))
plt.plot(epochs, [g[0][0] for g in gates], label='R0-H0')
plt.plot(epochs, [g[0][1] for g in gates], label='R0-H1')
plt.plot(epochs, [g[1][0] for g in gates], label='R1-H0')
plt.plot(epochs, [g[1][1] for g in gates], label='R1-H1')
plt.xlabel('Epoch')
plt.ylabel('Gate Value')
plt.legend()
plt.title('Regime Attention Gate Evolution')
plt.savefig('gate_evolution.png')
