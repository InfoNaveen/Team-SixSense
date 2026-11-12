# Stubs — Temporary Stand-ins for Teammate Modules

These stub files let the **Lead** develop and test `procurement/` and the
integration pipeline (`run_all.py`) *before* teammates deliver their modules.

Once the real implementations land, the stubs are no longer imported — the
pipeline automatically prefers the real module and only falls back to a stub on
`ImportError`.

## Stub Contracts

| Stub file | Replaces | Contract |
|---|---|---|
| `stub_fit_predict.py` | `models/train.py → fit_predict(X_train, y_train, X_test)` | Returns a DataFrame with columns `P10`, `P50`, `P90`.  P50 = mean(y_train); P10 = P50 × 0.85; P90 = P50 × 1.15. |

## Adding More Stubs

If you need to test against another teammate's module before it's ready, add a
stub here that honours the agreed contract (inputs & outputs) and returns
plausible dummy data.
