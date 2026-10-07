# Contributing

FlowCore is currently a research prototype.  Changes should preserve a strict
separation between:

1. mathematically exact scan kernels,
2. model/controller design,
3. experimental claims.

For any new operator family, add a sequential reference and a test proving that
parallel states and gradients agree within documented floating-point tolerance.
For any claimed compute improvement, report actual wall-clock/FLOP measurements;
do not infer compute sparsity from masked state alone.

Before opening a pull request:

```bash
python -m pip install -e . --no-build-isolation
pytest -q
```
